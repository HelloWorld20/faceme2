import os
import argparse
import json
from tqdm import tqdm

def main(args):
    pair_data = []

    if args.dataset_type == "FFHQ":
        for file in tqdm(os.listdir(args.ffhq_dir)):
            num = file.split('.')[-2]

            target_path = os.path.join(args.ffhq_dir , file)

            target_id_emb_path = os.path.join(args.ffhq_emb_dir, num + '.npy')
            target_clip_emb_path = os.path.join(args.ffhqref_emb_dir, num + '.npy')
            
            ref_id_emb_dir = os.path.join(args.ffhq_emb_dir, num)
            ref_clip_emb_dir = os.path.join(args.ffhqref_emb_dir, num)
            
            if not os.path.exists(ref_id_emb_dir):
                print(f"Warning: {ref_id_emb_dir} not found, skipping...")
                continue
            if not os.path.exists(ref_clip_emb_dir):
                print(f"Warning: {ref_clip_emb_dir} not found, skipping...")
                continue
            
            ref_id_emb_paths = [os.path.join(ref_id_emb_dir, f) for f in os.listdir(ref_id_emb_dir)]
            ref_clip_emb_paths = [os.path.join(ref_clip_emb_dir, f) for f in os.listdir(ref_clip_emb_dir)]
            
            ref_id_emb_paths = sorted(ref_id_emb_paths)
            ref_clip_emb_paths = sorted(ref_clip_emb_paths)

            pair_data.append(dict(target=target_path, target_emb=(target_id_emb_path, target_clip_emb_path), ref_emb=(ref_id_emb_paths, ref_clip_emb_paths)))
            
    elif args.dataset_type == "CelebHQRef":
        for person_id in tqdm(os.listdir(args.celebhq_dir)):
            person_dir = os.path.join(args.celebhq_dir, person_id)
            if not os.path.isdir(person_dir):
                continue
            
            ref_id_emb_dir = os.path.join(args.id_emb_dir, person_id)
            ref_clip_emb_dir = os.path.join(args.clip_emb_dir, person_id)
            
            if not os.path.exists(ref_id_emb_dir) or not os.path.exists(ref_clip_emb_dir):
                # Only warn if embeddings are expected but not found. 
                # Could be noisy if just iterating over random dirs, but acceptable.
                continue
                
            images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) < 2:
                # Need at least one target and one reference
                continue
                
            for target_img in images:
                target_path = os.path.join(person_dir, target_img)
                img_name_no_ext = target_img.split('.')[0]
                
                target_id_emb_path = os.path.join(ref_id_emb_dir, img_name_no_ext + '.npy')
                target_clip_emb_path = os.path.join(ref_clip_emb_dir, img_name_no_ext + '.npy')
                
                if not os.path.exists(target_id_emb_path) or not os.path.exists(target_clip_emb_path):
                    continue
                    
                ref_id_emb_paths = []
                ref_clip_emb_paths = []
                
                for ref_img in images:
                    if ref_img == target_img:
                        continue
                    ref_name_no_ext = ref_img.split('.')[0]
                    ref_id_path = os.path.join(ref_id_emb_dir, ref_name_no_ext + '.npy')
                    ref_clip_path = os.path.join(ref_clip_emb_dir, ref_name_no_ext + '.npy')
                    if os.path.exists(ref_id_path) and os.path.exists(ref_clip_path):
                        ref_id_emb_paths.append(ref_id_path)
                        ref_clip_emb_paths.append(ref_clip_path)
                
                if len(ref_id_emb_paths) == 0:
                    continue
                
                ref_id_emb_paths = sorted(ref_id_emb_paths)
                ref_clip_emb_paths = sorted(ref_clip_emb_paths)
                
                pair_data.append(dict(target=target_path, target_emb=(target_id_emb_path, target_clip_emb_path), ref_emb=(ref_id_emb_paths, ref_clip_emb_paths)))

    with open(os.path.join(args.save_dir, "train.json") , 'w') as f :
        for d in pair_data :
            json.dump(d , f)
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=["FFHQ", "CelebHQRef"], default="FFHQ", help="数据集类型")
    # For FFHQ
    parser.add_argument("--ffhq_dir", type=str, default="", help="FFHQ图像目录")
    parser.add_argument("--ffhq_emb_dir", type=str, default="", help="FFHQ ID嵌入目录 (output/id_emb/)")
    parser.add_argument("--ffhqref_emb_dir", type=str, default="", help="FFHQRef CLIP嵌入目录 (output/clip_emb/)")
    
    # For CelebHQRef
    parser.add_argument("--celebhq_dir", type=str, default="", help="CelebHQRefForRelease图像目录")
    parser.add_argument("--id_emb_dir", type=str, default="", help="ID嵌入目录")
    parser.add_argument("--clip_emb_dir", type=str, default="", help="CLIP嵌入目录")
    
    parser.add_argument("--save_dir", type=str, required=True, help="输出JSON保存目录")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir , exist_ok=True)

    main(args)