import argparse
import os

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="U-Net Model.")

        parser.add_argument("--set", type=str, default='val', help="Choose set.")
        # parser.add_argument("--data-label", type=str, default='./dataset/test/label', help="Path to the label folder.")
        parser.add_argument("--num_classes", type=int, default=1, help="The number of class.")
        # parser.add_argument("data_dir", type=str, default='./dataset/test/image', help="Path to the folder of the test data.")
        parser.add_argument("--restore-from", type=str, default='./snapshot/', help="Where restore model parameters from.")
        parser.add_argument("--save", type=str, default='./dataset/test/results', help="Path to save result.")
        parser.add_argument("--data-dir", type=str, default='./dataset', help="Path to the dataset root.")
        parser.add_argument("--is-label", type=bool, default=False, help="The test dataset have label or not.")
        return parser.parse_args()
