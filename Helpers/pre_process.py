import yaml


def pre_process( master_dict, nodes_dict):
    sequence = None
    image_path = None
    image_output_path = None
    image_crops = dict()
    output_EfficientHRNet = bool(master_dict['outputs']['EfficientHRNet'])

    for node_id, values in nodes_dict.items():
        print(node_id)
        image_path = values['image_path']
        sequence = values['sequence']
        image_output_path = values['image_output_path']
    return image_path, image_output_path, sequence

def open_multiple_yaml_files(tuple):
    for i in range(len(tuple)):
        tuple[i]=open_yaml_file(tuple[i])
    return tuple


def open_yaml_file(dict):
    # Open YAML file for Nodes
    stream = open(dict)
    dict = yaml.load(stream, Loader=yaml.SafeLoader)
    return dict