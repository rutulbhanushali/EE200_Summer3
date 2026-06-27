import argparse
import time
import pickle
import fingerprint as fp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", default="songs")
    parser.add_argument("--out", default="db.pkl")
    parser.add_argument("--sp_out", default="sp_index.pkl")
    args = parser.parse_args()

    print(f"Building fingerprint database from '{args.songs}'...")
    t0 = time.time()

    db = fp.FingerprintDB()
    n = db.build_from_folder(args.songs)

    if n == 0:
        print(f"WARNING: No audio files found in '{args.songs}/'.")
        return

    db.save(args.out)

    print(f"Building single-peak index from '{args.songs}'...")
    sp_index = fp.build_single_peak_index_from_folder(args.songs)
    with open(args.sp_out, "wb") as f:
        pickle.dump(sp_index, f)

    elapsed = time.time() - t0
    total_h = sum(db.songs.values())

    print(f"\nIndexed {n} song(s) | {total_h:,} total hashes")
    print(f"Saved DB to '{args.out}' and Single-Peak Index to '{args.sp_out}' ({elapsed:.1f}s)\n")
    print("Song breakdown:")
    for label, count in sorted(db.songs.items()):
        print(f"  {label:<40s}  {count:>8,} hashes")

if __name__ == "__main__":
    main()
