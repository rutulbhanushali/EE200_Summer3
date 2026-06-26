import fingerprint as fp
db = fp.FingerprintDB()
n = db.build_from_folder("songs")
db.save("db.pkl")
print(f"Indexed {n} songs, {sum(db.songs.values()):,} hashes -> db.pkl")

