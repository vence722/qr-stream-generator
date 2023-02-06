# %%
import qrcode
import zlib
import os
import tqdm
import imageio
import numpy as np
import hashlib
import base64
import shutil
config = {}
config["is_zip"] = True
config["chunk_str_size"] = 100
config["output_dir"] = "staging/stg_e/20230131_1058"
config["gif_duration"] = 0.1

# %%

if __name__ == '__main__':
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    file_name = 'test.zip'
    with open('sample/test.zip',mode='rb') as file:
        file_content = file.read()
        in_str = base64.b64encode(file_content).decode('ascii')

    hash_file_name = hashlib.md5(file_name.encode('utf-8')).hexdigest()
    chunk_str_size = config["chunk_str_size"]
    
    img_ls = []
    total_size = round(len(in_str) /  chunk_str_size + 0.5)
    for i in tqdm.tqdm(range(total_size)):
        _header = f"[{file_name}:{hash_file_name}:{i + 1}:{total_size}]"
        img = qrcode.make(_header + in_str[i * chunk_str_size : (i + 1) * chunk_str_size])
        img_path = f'{config["output_dir"]}/out_{i}.png'
        img.save(img_path)
        img_ls.append(imageio.imread(img_path))

    imageio.mimsave(f'{config["output_dir"]}/all.gif', img_ls, duration = config["gif_duration"])
    print('Done.')

