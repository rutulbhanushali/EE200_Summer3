import argparse
import time
import fingerprint as fp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", default="songs", help="Folder with audio files")
    parser.add_argument("--out", default="db.pkl", help="Output path")
    args = parser.parse_args()

    print(f"Building fingerprint database from '{args.songs}'...")
    t0 = time.time()

    db = fp.FingerprintDB()
    n = db.build_from_folder(args.songs)

    if n == 0:
        print(f"WARNING: No audio files found in '{args.songs}/'.")
        return

    db.save(args.out)
    elapsed = time.time() - t0
    total_h = sum(db.songs.values())

    print(f"Indexed {n} song(s) | {total_h:,} total hashes | Saved to '{args.out}' ({elapsed:.1f}s)\n")
    print("Song breakdown:")
    for label, count in sorted(db.songs.items()):
        print(f"  {label:<40s}  {count:>8,} hashes")

if __name__ == "__main__":
    main()