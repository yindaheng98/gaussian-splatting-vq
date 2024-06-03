import sqlite3


def load_colmap_cameras(load_cameras_path, dst_path):
    src_database = load_cameras_path + "/distorted/database.db"
    dst_database = dst_path + "/distorted/database.db"
    conn = sqlite3.connect(dst_database)
    c = conn.cursor()
    c.execute("DELETE FROM cameras")
    c.execute("DELETE FROM images")
    c.execute(f"ATTACH DATABASE '{src_database}' as 'other'")
    c.execute(f"INSERT INTO main.cameras SELECT * FROM other.cameras")
    c.execute(f"INSERT INTO main.images SELECT * FROM other.images")
    conn.commit()
    conn.close()
    print(src_database, "->", dst_database)
