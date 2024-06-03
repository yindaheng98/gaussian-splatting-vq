import sqlite3
import shutil
import os
import logging


def load_colmap_cameras(load_cameras_path, dst_path, colmap_command):
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
    mapper_input_path = dst_path + "/distorted/sparse/last_frame"
    if os.path.isdir(mapper_input_path):
        shutil.rmtree(mapper_input_path)
    os.makedirs(mapper_input_path)
    convert_cmd = colmap_command + " model_converter \
        --input_path " + load_cameras_path + "/distorted/sparse/0 \
        --output_path " + mapper_input_path + " \
        --output_type TXT"
    exit_code = os.system(convert_cmd)
    if exit_code != 0:
        logging.error(f"model_converter failed with code {exit_code}. Exiting.")
        exit(exit_code)
    open(mapper_input_path + "/images.txt", "w").close()
    open(mapper_input_path + "/points3D.txt", "w").close()
    return mapper_input_path
