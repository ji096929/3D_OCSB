"""Collect ScanNetV2 point clouds + semantic labels + superpoint ids.

Extended from COSeg preprocess/collect_scannet_data.py:
  - saves official .segs.json seg id as the 8th column
  - optional --split val/train/test to process only that split

Output format per scene: (N, 8)
    [:, 0:3] XYZ
    [:, 3:6] RGB
    [:, 6]   semantic label (0-19)
    [:, 7]   superpoint id (from _vh_clean_2.0.010000.segs.json)

room2blocks.py slices with boolean masks and keeps all columns, so the
8th column follows the points into every block automatically.
"""

import os
import sys
import json
import argparse
import numpy as np
from plyfile import PlyData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

CLASS_NAMES = []
RAW2SCANNET = {}


def get_raw2scannet_label_map(label_mapping_file):
    lines = [line.rstrip() for line in open(label_mapping_file)]
    lines = lines[1:]
    raw2scannet = {}
    label_classes_set = set(CLASS_NAMES)
    for i in range(len(lines)):
        elements = lines[i].split("\t")
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = "unannotated"
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


def read_ply_xyzrgb(filename):
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices


def collect_point_label(scene_path, scene_name, out_filename):
    # official over-segmentation: one segment id per vertex
    mesh_seg_filename = os.path.join(
        scene_path, "%s_vh_clean_2.0.010000.segs.json" % (scene_name)
    )
    with open(mesh_seg_filename) as jsondata:
        seg = np.array(json.load(jsondata)["segIndices"])

    segid_to_pointid = {}
    for i in range(len(seg)):
        segid_to_pointid.setdefault(int(seg[i]), []).append(i)

    ply_filename = os.path.join(scene_path, "%s_vh_clean_2.ply" % (scene_name))
    points = read_ply_xyzrgb(ply_filename)

    instance_segids = []
    labels = []
    annotation_filename = os.path.join(
        scene_path, "%s.aggregation.json" % (scene_name)
    )
    with open(annotation_filename) as jsondata:
        for x in json.load(jsondata)["segGroups"]:
            instance_segids.append(x["segments"])
            labels.append(x["label"])

    instance_points_list = []
    semantic_labels_list = []
    seg_ids_list = []
    for i in range(len(instance_segids)):
        pointids = []
        for segid in instance_segids[i]:
            pointids += segid_to_pointid[segid]
        pointids = np.array(pointids)
        instance_points = points[pointids, :]
        instance_points_list.append(instance_points)

        label = RAW2SCANNET.get(labels[i], "unannotated")
        label = CLASS_NAMES.index(label)
        semantic_labels_list.append(
            np.ones((instance_points.shape[0], 1)) * label
        )
        # keep the raw segment id for this instance's points
        seg_ids_list.append(seg[pointids][:, None])

    scene_points = np.concatenate(instance_points_list, 0)[:, 0:6]
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    seg_ids = np.concatenate(seg_ids_list, 0)
    data = np.concatenate((scene_points, semantic_labels, seg_ids), 1)
    np.save(out_filename, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="datasets/ScanNet/scans")
    parser.add_argument("--save_path", default="datasets/ScanNet")
    parser.add_argument(
        "--split", default="all", choices=["all", "val", "train", "test"]
    )
    parser.add_argument("--scene", default=None, help="process a single scene")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    SAVE_PATH = os.path.join(args.save_path, "scenes", "data")
    os.makedirs(SAVE_PATH, exist_ok=True)

    meta_path = os.path.join(ROOT_DIR, "datasets/ScanNet", "meta")
    CLASS_NAMES = [
        x.rstrip()
        for x in open(os.path.join(meta_path, "scannet_classnames.txt"))
    ]
    RAW2SCANNET = get_raw2scannet_label_map(
        os.path.join(meta_path, "scannetv2-labels.combined.tsv")
    )

    if args.scene:
        scene_names = [args.scene]
    else:
        scene_names = [
            o
            for o in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, o))
        ]
        if args.split != "all":
            split_file = os.path.join(meta_path, "scannetv2_%s.txt" % args.split)
            keep = set(x.strip() for x in open(split_file) if x.strip())
            scene_names = [n for n in scene_names if n in keep]

    print("%d scenes to process (split=%s)" % (len(scene_names), args.split))
    for scene_name in sorted(scene_names):
        out_filename = os.path.join(SAVE_PATH, scene_name + ".npy")
        if args.skip_existing and os.path.exists(out_filename):
            continue
        collect_point_label(
            os.path.join(DATA_PATH, scene_name), scene_name, out_filename
        )
    print("done")
