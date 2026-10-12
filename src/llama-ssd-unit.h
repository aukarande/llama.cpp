#pragma once

#include <cstdint>
#include <string>
#include <vector>

// a storage unit the RAM expert store reads a model from: one physical disk, or one volume that spans several disks (a
// stripe or a span reads from all of them at once). Its read speed is the unit's, whatever partition a file is on
struct llama_ssd_unit {
    std::string              id;     // disk-<16 hex> (from the disk's serial) or volume-<16 hex> (a multi-disk volume, or
                                     // one on a disk without a serial), stable across reboots and drive letters
    std::string              desc;   // for people: the disk's model or the volume, and where it is mounted
    std::vector<std::string> dirs;   // directories on it a scratch file may go to, in order of preference
};

// every fixed local unit with a mounted volume
std::vector<llama_ssd_unit> llama_ssd_units();
// the unit holding path, links followed ("" = unknown: not a local disk, or the system does not say)
std::string llama_ssd_unit_of(const std::string & path);

// a new file of random data in dir (created if missing), written past the system's file cache and removed when this
// is destroyed (on Windows also when the process dies)
struct llama_ssd_scratch {
    std::string path;
    uint64_t    bytes = 0;   // written

    llama_ssd_scratch() = default;
    llama_ssd_scratch(const llama_ssd_scratch &) = delete;
    llama_ssd_scratch & operator=(const llama_ssd_scratch &) = delete;

    // write up to max_bytes (a multiple of 16 MiB) for at most `seconds`; false: too little room or the file could not
    // be created (path empty), or it got fewer than min_bytes in that time (path set: the disk is too slow). Probe files
    // that dead processes left in dir are removed first
    bool create(const std::string & dir, uint64_t max_bytes, uint64_t min_bytes, double seconds);
    ~llama_ssd_scratch();

private:
    std::string made_dir;    // the directory create() made, removed with the file
#ifdef _WIN32
    void *      handle = nullptr;
#endif
};
