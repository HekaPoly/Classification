


names = [
    "shoulder_abduction_",
    "shoulder_adduction_",
    "shoulder_flexion_",
    "shoulder_extension_",
    "elbow_extension_",
    "elbow_flexion_",
    "rest_low_arm_",
    "rest_post_elbow_flexion_",
    "rest_post_shoulder_flexion_",
    "rest_post_abduction_"
]


def to_string(class_value):
    return names[class_value]


def from_string(class_name):
    return names.index(class_name)


def get_num_classes():
    return len(names)
