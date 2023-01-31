# %%
import qrcode
import zlib
import os
import tqdm
import imageio
import numpy as np
config = {}
config["is_zip"] = False
config["chunk_str_size"] = 100
config["output_dir"] = "staging/stg_e/20230131_1058"
config["gif_duration"] = 0.5

# %%


if __name__ == '__main__':
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])


    in_str = '''
        def get_tr_iter(train_data, FLAGS, NodeMinibatchIterator):
            G = train_data[0]
            features = train_data[1]
            id_map = train_data[2]
            class_map  = train_data[4]
            if isinstance(list(class_map.values())[0], list):
                num_classes = len(list(class_map.values())[0])
            else:
                num_classes = len(set(class_map.values()))
            context_pairs = train_data[3] if FLAGS.random_context else None
            placeholders = {
                'labels' : tf.cast(tf.compat.v1.distributions.Bernoulli(probs=0.7).sample(sample_shape=(1, num_classes)), tf.float32),
                'batch' : tf.constant(list(G.nodes)[:1], dtype=tf.int32, name='batch1'),
                'dropout': tf.constant(0., dtype=tf.float32, name='batch1'),
                'batch_size' : tf.constant(FLAGS.batch_size, dtype=tf.float32, name='batch1'),
            }
            minibatch = NodeMinibatchIterator(G,
                    id_map,
                    placeholders, 
                    class_map,
                    num_classes,
                    batch_size=FLAGS.batch_size,
                    max_degree=FLAGS.max_degree, 
                    context_pairs = context_pairs)
            return minibatch
        '''
    if config["is_zip"]:
        in_str = zlib.compress(in_str.encode('utf-8'))

    
    chunk_str_size = config["chunk_str_size"]
    img_ls = []
    for i in tqdm.tqdm(range(round(len(in_str) /  chunk_str_size + 0.5))):
        img = qrcode.make(f"[TIME_TOKEN_{i}]\n" + in_str[i * chunk_str_size : (i + 1) * chunk_str_size])
        img_path = f'{config["output_dir"]}/out_{i}.png'
        img.save(img_path)
        img_ls.append(imageio.imread(img_path))

    imageio.mimsave(f'{config["output_dir"]}/all.gif', img_ls, duration = config["gif_duration"])
    print('Done.')

