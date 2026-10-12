#pragma once

#include "llama-impl.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#else
#   include <fcntl.h>
#   include <unistd.h>
#endif

// unbuffered reads of the model's files for the SSD tier: offsets, lengths and buffers are sector aligned. Windows
// keeps every read of a batch in flight on one completion port; elsewhere the reads run one after another
struct llama_ssd_reader {
    static constexpr size_t sector = 4096;
#ifdef _WIN32
    static constexpr bool overlap = true;   // the reads of a batch are in flight together
    struct request {
        OVERLAPPED ov;
        size_t     need;   // bytes the caller uses: a read cut short at the end of the file may stop after them
    };
    std::vector<HANDLE> files;
    HANDLE port = nullptr;
    std::vector<std::unique_ptr<request>> reqs;
    size_t n_used  = 0;
    size_t pending = 0;
    bool   failed  = false;
    DWORD  last_error = 0;

    ~llama_ssd_reader() {
        for (HANDLE h : files) {
            CloseHandle(h);
        }
        if (port != nullptr) {
            CloseHandle(port);
        }
    }
    // share_all: let others write and delete the files too (a scratch file held open for writing and delete on close);
    // else a model file cannot be replaced while it is read
    bool open(const std::vector<std::string> & paths, bool share_all = false) {
        for (size_t i = 0; i < paths.size(); i++) {
            // the paths are UTF-8, as ggml_fopen takes them
            const int wlen = MultiByteToWideChar(CP_UTF8, 0, paths[i].c_str(), -1, nullptr, 0);
            std::wstring wpath(wlen > 0 ? (size_t) wlen : 0, L'\0');
            if (wlen <= 0 || MultiByteToWideChar(CP_UTF8, 0, paths[i].c_str(), -1, wpath.data(), wlen) <= 0) {
                LLAMA_LOG_ERROR("%s: cannot convert %s to UTF-16\n", __func__, paths[i].c_str());
                return false;
            }
            HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | (share_all ? FILE_SHARE_WRITE | FILE_SHARE_DELETE : 0),
                nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
            if (h == INVALID_HANDLE_VALUE) {
                LLAMA_LOG_ERROR("%s: cannot open %s for unbuffered reads (error %lu)\n", __func__, paths[i].c_str(), GetLastError());
                return false;
            }
            files.push_back(h);
            HANDLE p = CreateIoCompletionPort(h, port, (ULONG_PTR) i, 1);
            if (p == nullptr) {
                LLAMA_LOG_ERROR("%s: cannot attach %s to the completion port (error %lu)\n", __func__, paths[i].c_str(), GetLastError());
                return false;
            }
            port = p;
        }
        return true;
    }
    // false: the read was not issued (the system takes no more reads in flight); wait_all does not count it
    bool try_read(size_t split, uint64_t off, size_t len, void * dst, size_t need) {
        if (n_used == reqs.size()) {
            reqs.push_back(std::make_unique<request>());
        }
        request * r = reqs[n_used].get();
        memset(&r->ov, 0, sizeof(r->ov));
        r->ov.Offset     = (DWORD) (off & 0xffffffffu);
        r->ov.OffsetHigh = (DWORD) (off >> 32);
        r->need = need;
        if (!ReadFile(files[split], dst, (DWORD) len, nullptr, &r->ov)) {
            last_error = GetLastError();
            if (last_error != ERROR_IO_PENDING) {
                return false;
            }
        }
        n_used++;
        pending++;
        return true;
    }
    void read(size_t split, uint64_t off, size_t len, void * dst, size_t need) {
        if (!try_read(split, off, len, dst, need)) {
            LLAMA_LOG_ERROR("%s: read of %zu bytes at %llu failed (error %lu)\n", __func__, len, (unsigned long long) off, last_error);
            failed = true;
        }
    }
    bool wait_all() {
        OVERLAPPED_ENTRY done[64];
        while (pending > 0) {
            ULONG n = 0;
            if (!GetQueuedCompletionStatusEx(port, done, 64, &n, INFINITE, FALSE)) {
                failed = true;
                break;
            }
            for (ULONG i = 0; i < n; i++) {
                const request * r = CONTAINING_RECORD(done[i].lpOverlapped, request, ov);
                if (done[i].lpOverlapped->Internal != 0 || done[i].dwNumberOfBytesTransferred < r->need) {
                    if (!failed) {
                        LLAMA_LOG_ERROR("%s: read at %llu failed: status 0x%08llx, %lu of %zu bytes\n", __func__,
                            ((unsigned long long) r->ov.OffsetHigh << 32) | r->ov.Offset,
                            (unsigned long long) done[i].lpOverlapped->Internal,
                            (unsigned long) done[i].dwNumberOfBytesTransferred, r->need);
                    }
                    failed = true;
                }
            }
            pending -= n;
        }
        n_used = 0;
        const bool ok = !failed;
        failed = false;
        return ok;
    }
#else
    static constexpr bool overlap = false;
    std::vector<int> files;
    bool failed = false;

    ~llama_ssd_reader() {
        for (int fd : files) {
            close(fd);
        }
    }
    bool open(const std::vector<std::string> & paths, bool share_all = false) {
        (void) share_all;
        for (const auto & p : paths) {
            int fd = -1;
#ifdef O_DIRECT
            fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
#endif
            if (fd < 0) {
                // a file system without direct reads: the page cache stays in the path
                fd = ::open(p.c_str(), O_RDONLY);
            }
            if (fd < 0) {
                LLAMA_LOG_ERROR("%s: cannot open %s\n", __func__, p.c_str());
                return false;
            }
#ifdef F_NOCACHE
            fcntl(fd, F_NOCACHE, 1);
#endif
            files.push_back(fd);
        }
        return true;
    }
    void read(size_t split, uint64_t off, size_t len, void * dst, size_t need) {
        size_t got = 0;
        while (got < len) {
            const ssize_t r = pread(files[split], (char *) dst + got, len - got, (off_t) (off + got));
            if (r <= 0) {
                break;
            }
            got += (size_t) r;
        }
        failed = failed || got < need;
    }
    bool try_read(size_t split, uint64_t off, size_t len, void * dst, size_t need) {
        read(split, off, len, dst, need);
        return true;
    }
    bool wait_all() {
        const bool ok = !failed;
        failed = false;
        return ok;
    }
#endif
};
