#include "llama-ssd-unit.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>

#ifdef _WIN32
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#   include <winioctl.h>
#else
#   include <cerrno>
#   include <dirent.h>
#   include <fcntl.h>
#   include <signal.h>
#   include <limits.h>
#   include <sys/stat.h>
#   include <sys/statvfs.h>
#   include <unistd.h>
#   ifdef __linux__
#       include <sys/sysmacros.h>
#   endif
#endif

static std::string ssd_unit_id(const char * kind, const std::string & identity) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const unsigned char c : identity) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%s-%016llx", kind, (unsigned long long) h);
    return buf;
}

static std::string ssd_trim(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    const size_t e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

namespace {
struct ssd_vol_unit {
    std::string id;
    std::string desc;
};
}

// one profile line per unit: the description names the model and the first two mount points only, on one line
static void ssd_add_dir(llama_ssd_unit & u, int & n_shown, const std::string & dir, bool mount) {
    if (std::find(u.dirs.begin(), u.dirs.end(), dir) != u.dirs.end()) {
        return;
    }
    u.dirs.push_back(dir);
    if (mount && n_shown++ < 2) {
        std::string d = dir;
        for (char & c : d) {
            c = (unsigned char) c < 0x20 ? '?' : c;
        }
        u.desc += " " + d;
    }
}

// the scratch data: random bytes with each 4 KiB page stamped with its index, nothing for a drive to compress or share
namespace {
struct ssd_scratch_buf {
    static constexpr size_t chunk = 16u << 20;
    std::vector<char> raw = std::vector<char>(chunk + 4096);
    char * buf = (char *) (((uintptr_t) raw.data() + 4095) & ~(uintptr_t) 4095);

    ssd_scratch_buf() {
        std::mt19937_64 rng(std::random_device{}());
        for (size_t i = 0; i < chunk; i += sizeof(uint64_t)) {
            const uint64_t x = rng();
            memcpy(buf + i, &x, sizeof(x));
        }
    }
    void stamp(uint64_t off) {
        for (size_t p = 0; p < chunk; p += 4096) {
            const uint64_t page = (off + p) / 4096;
            memcpy(buf + p, &page, sizeof(page));
        }
    }
};
}

// the bytes to write: half of what is free past the minimum, in whole chunks, at most max_bytes (0 = too little room)
static uint64_t ssd_scratch_bytes(uint64_t free_b, uint64_t max_bytes, uint64_t min_bytes) {
    const uint64_t chunk = ssd_scratch_buf::chunk;
    const uint64_t room  = free_b > min_bytes ? (free_b - min_bytes) / 2 / chunk * chunk : 0;
    const uint64_t b     = std::min(max_bytes, room);
    return b >= min_bytes ? b : 0;
}

static std::string ssd_scratch_name(unsigned long pid) {
    std::mt19937_64 rng(std::random_device{}());
    char buf[96];
    snprintf(buf, sizeof(buf), "llama-ssd-probe-%lu-%016llx.tmp", pid, (unsigned long long) rng());
    return buf;
}

// the process that named a probe file (0 = not one)
static unsigned long ssd_scratch_pid(const char * name) {
    unsigned long pid = 0;
    char hex[17] = {};
    char ext[5]  = {};
    return sscanf(name, "llama-ssd-probe-%lu-%16[0-9a-f].%4s", &pid, hex, ext) == 3 && strcmp(ext, "tmp") == 0 ? pid : 0;
}

static std::string ssd_join(const std::string & dir, const std::string & name) {
    const char last = dir.empty() ? '/' : dir.back();
    return dir + (last == '/' || last == '\\' ? "" : "/") + name;
}

#ifdef _WIN32

static std::string ssd_utf8(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) {
        return "";
    }
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    s.pop_back();
    return s;
}

static std::wstring ssd_wide(const std::string & s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 1) {
        return L"";
    }
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    w.pop_back();
    return w;
}

// the physical disks under a volume (\\?\Volume{...}\)
static std::vector<DWORD> ssd_volume_disks(const std::wstring & vol) {
    std::wstring dev = vol;
    if (!dev.empty() && dev.back() == L'\\') {
        dev.pop_back();   // the volume device, not its root directory
    }
    std::vector<DWORD> disks;
    HANDLE h = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return disks;
    }
    std::vector<char> buf(sizeof(VOLUME_DISK_EXTENTS) + 15 * sizeof(DISK_EXTENT));
    DWORD got = 0;
    BOOL ok = FALSE;
    while (!(ok = DeviceIoControl(h, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS, nullptr, 0, buf.data(), (DWORD) buf.size(), &got, nullptr)) &&
            GetLastError() == ERROR_MORE_DATA && buf.size() < (1u << 20)) {
        buf.resize(buf.size() * 2);
    }
    CloseHandle(h);
    if (!ok) {
        return disks;
    }
    const auto * ext = (const VOLUME_DISK_EXTENTS *) buf.data();
    for (DWORD i = 0; i < ext->NumberOfDiskExtents; i++) {
        disks.push_back(ext->Extents[i].DiskNumber);
    }
    std::sort(disks.begin(), disks.end());
    disks.erase(std::unique(disks.begin(), disks.end()), disks.end());
    return disks;
}

// vendor, product and serial of a physical disk ("" = it gives no serial); product: its model name
static std::string ssd_disk_identity(DWORD disk, std::string & product) {
    wchar_t path[64];
    swprintf(path, 64, L"\\\\.\\PhysicalDrive%lu", (unsigned long) disk);
    HANDLE h = CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return "";
    }
    STORAGE_PROPERTY_QUERY q = {};
    q.PropertyId = StorageDeviceProperty;
    q.QueryType  = PropertyStandardQuery;
    std::vector<char> buf(4096);
    DWORD got = 0;
    const BOOL ok = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf.data(), (DWORD) buf.size(), &got, nullptr);
    CloseHandle(h);
    if (!ok || got < sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
        return "";
    }
    const auto * d = (const STORAGE_DEVICE_DESCRIPTOR *) buf.data();
    auto field = [&](DWORD off) -> std::string {
        return off == 0 || off >= got ? "" : ssd_trim(std::string(buf.data() + off, strnlen(buf.data() + off, got - off)));
    };
    product = field(d->ProductIdOffset);
    const std::string serial = field(d->SerialNumberOffset);
    return serial.empty() ? "" : field(d->VendorIdOffset) + "|" + product + "|" + serial;
}

static bool ssd_unit_of_volume(const std::wstring & vol, ssd_vol_unit & u) {
    const auto disks = ssd_volume_disks(vol);
    if (disks.empty()) {
        return false;
    }
    std::string product;
    const std::string ident = disks.size() == 1 ? ssd_disk_identity(disks[0], product) : "";
    if (!ident.empty()) {
        u.id   = ssd_unit_id("disk", ident);
        u.desc = product;
        return true;
    }
    // several disks, or a disk without a serial (its number can change when disks come and go): the volume's own id
    u.id   = ssd_unit_id("volume", ssd_utf8(vol.c_str()));
    u.desc = disks.size() == 1 ? (product.empty() ? "disk " + std::to_string(disks[0]) : product) : "volume over disks";
    for (size_t i = 0; disks.size() > 1 && i < disks.size(); i++) {
        u.desc += (i == 0 ? " " : ",") + std::to_string(disks[i]);
    }
    return true;
}

// the volume (\\?\Volume{...}\) holding path: the final path with links and junctions followed, else from the path text
static std::wstring ssd_volume_of(const std::string & path) {
    const std::wstring w = ssd_wide(path);
    if (w.empty()) {
        return L"";
    }
    HANDLE h = CreateFileW(w.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        std::vector<wchar_t> buf(1024);
        DWORD n = GetFinalPathNameByHandleW(h, buf.data(), (DWORD) buf.size(), VOLUME_NAME_GUID);
        if (n >= buf.size()) {
            buf.resize((size_t) n + 1);
            n = GetFinalPathNameByHandleW(h, buf.data(), (DWORD) buf.size(), VOLUME_NAME_GUID);
        }
        CloseHandle(h);
        const std::wstring f(buf.data(), n > 0 && n < buf.size() ? n : 0);
        const size_t e = f.find(L'\\', 4);   // \\?\Volume{...}\ ends at the backslash after the braces
        // no volume name: the target is not on a local volume (a network share behind a link)
        return f.compare(0, 11, L"\\\\?\\Volume{") == 0 && e != std::wstring::npos ? f.substr(0, e + 1) : L"";
    }
    const DWORD n_full = GetFullPathNameW(w.c_str(), 0, nullptr, nullptr);
    if (n_full == 0) {
        return L"";
    }
    std::vector<wchar_t> full((size_t) n_full + 1);
    std::vector<wchar_t> mount((size_t) n_full + 1);
    wchar_t vol[MAX_PATH + 1];
    if (GetFullPathNameW(w.c_str(), (DWORD) full.size(), full.data(), nullptr) == 0 ||
            !GetVolumePathNameW(full.data(), mount.data(), (DWORD) mount.size()) ||
            !GetVolumeNameForVolumeMountPointW(mount.data(), vol, MAX_PATH + 1)) {
        return L"";
    }
    return vol;
}

std::string llama_ssd_unit_of(const std::string & path) {
    const std::wstring vol = ssd_volume_of(path);
    ssd_vol_unit u;
    return !vol.empty() && ssd_unit_of_volume(vol, u) ? u.id : "";
}

std::vector<llama_ssd_unit> llama_ssd_units() {
    // first on their units: directories this user can write to (a volume root may take folders only)
    std::vector<std::pair<std::string, std::string>> pref;   // unit, dir
    wchar_t buf[MAX_PATH + 1];
    if (GetTempPathW(MAX_PATH + 1, buf) > 0) {
        pref.push_back({ "", ssd_utf8(buf) });
    }
    if (GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH + 1) > 0) {
        pref.push_back({ "", ssd_utf8(buf) });
    }
    if (GetCurrentDirectoryW(MAX_PATH + 1, buf) > 0) {
        pref.push_back({ "", ssd_utf8(buf) });
    }
    for (auto & p : pref) {
        p.first = llama_ssd_unit_of(p.second);
    }
    std::vector<llama_ssd_unit> units;
    std::vector<int> shown;
    std::map<std::string, size_t> at;
    wchar_t vol[MAX_PATH + 1];
    HANDLE f = FindFirstVolumeW(vol, MAX_PATH + 1);
    if (f == INVALID_HANDLE_VALUE) {
        return units;
    }
    do {
        if (GetDriveTypeW(vol) != DRIVE_FIXED) {
            continue;
        }
        // its mount points (none: nowhere to put a file)
        std::vector<wchar_t> names(1024);
        DWORD len = 0;
        if (!GetVolumePathNamesForVolumeNameW(vol, names.data(), (DWORD) names.size(), &len)) {
            if (GetLastError() != ERROR_MORE_DATA) {
                continue;
            }
            names.resize(len);
            if (!GetVolumePathNamesForVolumeNameW(vol, names.data(), (DWORD) names.size(), &len)) {
                continue;
            }
        }
        std::vector<std::string> mounts;
        for (const wchar_t * p = names.data(); *p; p += wcslen(p) + 1) {
            mounts.push_back(ssd_utf8(p));
        }
        ssd_vol_unit u;
        if (mounts.empty() || !ssd_unit_of_volume(vol, u)) {
            continue;
        }
        auto it = at.find(u.id);
        if (it == at.end()) {
            it = at.emplace(u.id, units.size()).first;
            units.push_back({ u.id, u.desc, {} });
            shown.push_back(0);
            for (const auto & p : pref) {
                if (p.first == u.id) {
                    ssd_add_dir(units.back(), shown.back(), p.second, false);
                }
            }
        }
        for (const auto & m : mounts) {
            ssd_add_dir(units[it->second], shown[it->second], m, true);
        }
    } while (FindNextVolumeW(f, vol, MAX_PATH + 1));
    FindVolumeClose(f);
    // a mount point's root may refuse files but take a folder: the scratch file goes into one, made and removed
    for (auto & unit : units) {
        const size_t n = unit.dirs.size();
        for (size_t i = 0; i < n; i++) {
            const std::string d = unit.dirs[i];
            const bool is_pref = std::find_if(pref.begin(), pref.end(), [&](const auto & p) { return p.second == d; }) != pref.end();
            if (!is_pref && d.size() >= 2 && d.back() == '\\') {
                unit.dirs.push_back(d + "llama-ssd-probe");
            }
        }
    }
    return units;
}

bool llama_ssd_scratch::create(const std::string & dir, uint64_t max_bytes, uint64_t min_bytes, double seconds) {
    const std::wstring wdir = ssd_wide(dir);
    if (wdir.empty()) {
        return false;
    }
    if (GetFileAttributesW(wdir.c_str()) == INVALID_FILE_ATTRIBUTES) {
        if (!CreateDirectoryW(wdir.c_str(), nullptr)) {
            return false;
        }
        made_dir = dir;
    }
    // a reset leaves a probe file behind (delete on close does not outlive the system): those of dead processes go
    WIN32_FIND_DATAW fd;
    HANDLE fh = FindFirstFileW(ssd_wide(ssd_join(dir, "llama-ssd-probe-*.tmp")).c_str(), &fd);
    for (BOOL more = fh != INVALID_HANDLE_VALUE; more; more = FindNextFileW(fh, &fd)) {
        const unsigned long pid = ssd_scratch_pid(ssd_utf8(fd.cFileName).c_str());
        HANDLE ph = pid != 0 ? OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD) pid) : nullptr;
        if (ph != nullptr) {
            CloseHandle(ph);
        } else if (pid != 0 && GetLastError() == ERROR_INVALID_PARAMETER) {
            DeleteFileW(ssd_wide(ssd_join(dir, ssd_utf8(fd.cFileName))).c_str());
        }
    }
    if (fh != INVALID_HANDLE_VALUE) {
        FindClose(fh);
    }
    ULARGE_INTEGER avail = {};
    const uint64_t want = GetDiskFreeSpaceExW(wdir.c_str(), &avail, nullptr, nullptr) ? ssd_scratch_bytes(avail.QuadPart, max_bytes, min_bytes) : 0;
    path = ssd_join(dir, ssd_scratch_name(GetCurrentProcessId()));
    // a new name only (no file or link of that name is followed), deleted when the last handle closes - the process
    // dying included; readers open it with every sharing mode
    HANDLE h = want == 0 ? INVALID_HANDLE_VALUE : CreateFileW(ssd_wide(path).c_str(), GENERIC_WRITE | DELETE, FILE_SHARE_READ | FILE_SHARE_DELETE,
        nullptr, CREATE_NEW, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        if (!made_dir.empty()) {
            RemoveDirectoryW(wdir.c_str());
            made_dir.clear();
        }
        path.clear();
        return false;
    }
    handle = h;
    ssd_scratch_buf sb;
    const auto t0 = std::chrono::steady_clock::now();
    while (bytes < want && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        sb.stamp(bytes);
        DWORD put = 0;
        if (!WriteFile(h, sb.buf, (DWORD) sb.chunk, &put, nullptr) || put != sb.chunk) {
            break;
        }
        bytes += sb.chunk;
    }
    return bytes >= min_bytes;
}

llama_ssd_scratch::~llama_ssd_scratch() {
    if (handle != nullptr) {
        CloseHandle((HANDLE) handle);
    }
    if (!made_dir.empty()) {
        RemoveDirectoryW(ssd_wide(made_dir).c_str());
    }
}

#else

#ifdef __linux__

static std::string ssd_read_first_line(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return "";
    }
    char buf[256] = {};
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    return ok ? ssd_trim(buf) : "";
}

static std::string ssd_realpath(const std::string & path) {
    char buf[PATH_MAX];
    return realpath(path.c_str(), buf) != nullptr ? buf : "";
}

static bool ssd_exists(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string ssd_parent(const std::string & dir) {
    return dir.substr(0, dir.rfind('/'));
}

// the physical disks under a block device's sysfs directory: a partition's disk, a mapper or md device's leaves (a
// partitioned array through its parent), the disk itself. Devices without hardware behind them (loop, ram, zram) and
// removable media give none
static void ssd_leaf_disks(const std::string & dir, std::set<std::string> & out, int depth = 0) {
    if (dir.empty() || depth > 8) {
        return;
    }
    if (ssd_exists(dir + "/partition")) {
        ssd_leaf_disks(ssd_parent(dir), out, depth + 1);
        return;
    }
    std::vector<std::string> slaves;
    if (DIR * d = opendir((dir + "/slaves").c_str())) {
        while (const dirent * e = readdir(d)) {
            if (e->d_name[0] != '.') {
                slaves.push_back(e->d_name);
            }
        }
        closedir(d);
    }
    if (!slaves.empty()) {
        for (const auto & s : slaves) {
            ssd_leaf_disks(ssd_realpath(dir + "/slaves/" + s), out, depth + 1);
        }
        return;
    }
    if (ssd_exists(dir + "/device") && ssd_read_first_line(dir + "/removable") != "1") {
        out.insert(dir);
    }
}

// model and serial of a disk ("" = it gives no serial)
static std::string ssd_disk_identity(const std::string & disk, std::string & product) {
    product = ssd_read_first_line(disk + "/device/model");
    for (const char * f : { "/wwid", "/device/wwid", "/device/serial" }) {
        const std::string s = ssd_read_first_line(disk + f);
        if (!s.empty()) {
            return product + "|" + s;
        }
    }
    return "";
}

static std::string ssd_unescape(const std::string & s);

// the block device behind a path whose st_dev is not one (a btrfs subvolume reports its own anonymous device): the
// source of the mount that holds it - the longest mount point in /proc/self/mountinfo that prefixes its real path
static bool ssd_mount_source_dev(const std::string & path, dev_t & out) {
    const std::string real = ssd_realpath(path);
    FILE * f = real.empty() ? nullptr : fopen("/proc/self/mountinfo", "r");
    if (f == nullptr) {
        return false;
    }
    char line[8192];
    size_t best = 0;
    std::string src_best;
    while (fgets(line, sizeof(line), f) != nullptr) {
        char mnt[4096];
        const char * sep = strstr(line, " - ");
        char fstype[256];
        char src[4096];
        if (sscanf(line, "%*s %*s %*s %*s %4095s", mnt) != 1 || sep == nullptr || sscanf(sep + 3, "%255s %4095s", fstype, src) != 2) {
            continue;
        }
        const std::string m = ssd_unescape(mnt);
        const bool under = m == "/" || (real.compare(0, m.size(), m) == 0 && (real.size() == m.size() || real[m.size()] == '/'));
        if (under && m.size() >= best) {   // equal: the later mount covers the earlier
            best     = m.size();
            src_best = ssd_unescape(src);
        }
    }
    fclose(f);
    struct stat st;
    if (src_best.compare(0, 5, "/dev/") != 0 || stat(src_best.c_str(), &st) != 0 || !S_ISBLK(st.st_mode)) {
        return false;
    }
    out = st.st_rdev;
    return true;
}

// a name for a partition or volume that survives replugging: its partition uuid, else its file system uuid
static std::string ssd_persistent_name(const std::string & name) {
    for (const char * dir : { "/dev/disk/by-partuuid", "/dev/disk/by-uuid" }) {
        DIR * d = opendir(dir);
        if (d == nullptr) {
            continue;
        }
        std::string found;
        while (const dirent * e = readdir(d)) {
            if (e->d_name[0] != '.' && ssd_realpath(std::string(dir) + "/" + e->d_name) == "/dev/" + name) {
                found = e->d_name;
                break;
            }
        }
        closedir(d);
        if (!found.empty()) {
            return found;
        }
    }
    return name;
}

static bool ssd_unit_of_path(const std::string & path, ssd_vol_unit & u) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return false;
    }
    dev_t dev = st.st_dev;
    if (major(dev) == 0 && !ssd_mount_source_dev(path, dev)) {
        return false;   // tmpfs, network and other file systems without a block device
    }
    char p[64];
    snprintf(p, sizeof(p), "/sys/dev/block/%u:%u", major(dev), minor(dev));
    const std::string dir = ssd_realpath(p);
    std::set<std::string> disks;
    ssd_leaf_disks(dir, disks);
    if (disks.size() == 1) {
        std::string product;
        const std::string & d     = *disks.begin();
        const std::string   ident = ssd_disk_identity(d, product);
        const std::string   name  = d.substr(d.rfind('/') + 1);
        if (!ident.empty()) {
            u.id   = ssd_unit_id("disk", ident);
            u.desc = product.empty() ? name : product;
            return true;
        }
    } else if (disks.empty()) {
        return false;
    }
    // several disks, or a disk without a serial: the volume, by its array or mapper uuid when it has one
    const std::string vdir = ssd_exists(dir + "/partition") ? ssd_parent(dir) : dir;
    std::string ident = ssd_read_first_line(vdir + "/dm/uuid");
    if (ident.empty()) {
        ident = ssd_read_first_line(vdir + "/md/uuid");
    }
    const std::string name = dir.substr(dir.rfind('/') + 1);
    u.id   = ssd_unit_id("volume", ident.empty() ? ssd_persistent_name(name) : ident);
    u.desc = disks.size() == 1 ? name : "volume " + name + " over " + std::to_string(disks.size()) + " disks";
    return true;
}

// a mount point as /proc/self/mounts escapes it (\040 for a space)
static std::string ssd_unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 3 < s.size() && isdigit((unsigned char) s[i + 1]) && isdigit((unsigned char) s[i + 2]) &&
                isdigit((unsigned char) s[i + 3])) {
            out += (char) ((s[i + 1] - '0') * 64 + (s[i + 2] - '0') * 8 + (s[i + 3] - '0'));
            i += 3;
        } else {
            out += s[i];
        }
    }
    return out;
}

std::string llama_ssd_unit_of(const std::string & path) {
    ssd_vol_unit u;
    return ssd_unit_of_path(path, u) ? u.id : "";
}

std::vector<llama_ssd_unit> llama_ssd_units() {
    std::vector<llama_ssd_unit> units;
    std::vector<int> shown;
    std::map<std::string, size_t> at;
    auto add = [&](const std::string & dir, bool mount) {
        ssd_vol_unit u;
        if (dir.empty() || !ssd_unit_of_path(dir, u)) {
            return;
        }
        auto it = at.find(u.id);
        if (it == at.end()) {
            it = at.emplace(u.id, units.size()).first;
            units.push_back({ u.id, u.desc, {} });
            shown.push_back(0);
        }
        ssd_add_dir(units[it->second], shown[it->second], dir, mount);
    };
    // first on their units: directories this user can write to (a mount point's root may be root's only)
    const char * tmp = getenv("TMPDIR");
    add(tmp != nullptr && *tmp ? tmp : "/tmp", false);
    add("/var/tmp", false);
    if (const char * home = getenv("HOME")) {
        add(home, false);
    }
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        add(cwd, false);
    }
    if (FILE * f = fopen("/proc/self/mounts", "r")) {
        char src[4096];
        char mnt[4096];
        while (fscanf(f, "%4095s %4095s%*[^\n]", src, mnt) == 2) {
            if (strncmp(src, "/dev/", 5) == 0) {
                add(ssd_unescape(mnt), true);
            }
        }
        fclose(f);
    }
    return units;
}

#else

// no unit lookup on this system: the store's reads stay unpriced
std::string llama_ssd_unit_of(const std::string & path) {
    (void) path;
    return "";
}

std::vector<llama_ssd_unit> llama_ssd_units() {
    return {};
}

#endif

bool llama_ssd_scratch::create(const std::string & dir, uint64_t max_bytes, uint64_t min_bytes, double seconds) {
    struct stat st;
    if (stat(dir.c_str(), &st) != 0) {
        if (mkdir(dir.c_str(), 0700) != 0) {
            return false;
        }
        made_dir = dir;
    }
    // a killed run leaves its probe file behind: those of dead processes go
    if (DIR * d = opendir(dir.c_str())) {
        while (const dirent * e = readdir(d)) {
            const unsigned long pid = ssd_scratch_pid(e->d_name);
            if (pid != 0 && kill((pid_t) pid, 0) != 0 && errno == ESRCH) {
                unlink(ssd_join(dir, e->d_name).c_str());
            }
        }
        closedir(d);
    }
    struct statvfs vfs;
    const uint64_t want = statvfs(dir.c_str(), &vfs) == 0 ? ssd_scratch_bytes((uint64_t) vfs.f_bavail * vfs.f_frsize, max_bytes, min_bytes) : 0;
    path = ssd_join(dir, ssd_scratch_name((unsigned long) getpid()));
    // a new name only: no file or link of that name is followed. Direct writes where the file system takes them
    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = want > 0 ? open(path.c_str(), flags, 0600) : -1;
#ifdef O_DIRECT
    if (fd >= 0) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_DIRECT);
    }
#endif
    if (fd < 0) {
        if (!made_dir.empty()) {
            rmdir(made_dir.c_str());
            made_dir.clear();
        }
        path.clear();
        return false;
    }
#ifdef F_NOCACHE
    fcntl(fd, F_NOCACHE, 1);
#endif
    ssd_scratch_buf sb;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    while (ok && bytes < want && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        sb.stamp(bytes);
        size_t put = 0;
        while (ok && put < sb.chunk) {
            const ssize_t r = pwrite(fd, sb.buf + put, sb.chunk - put, (off_t) (bytes + put));
            ok = r > 0;
            put += ok ? (size_t) r : 0;
        }
        bytes += ok ? sb.chunk : 0;
    }
    fsync(fd);
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    close(fd);
    return bytes >= min_bytes;
}

llama_ssd_scratch::~llama_ssd_scratch() {
    if (!path.empty()) {
        unlink(path.c_str());
    }
    if (!made_dir.empty()) {
        rmdir(made_dir.c_str());
    }
}

#endif
