from maf_experiments import model
import os
import json


def load_model(fn, save_dir="models"):
    args = {}
    old_args = save_dir + '/' + fn + '_args.txt'
    old_path = save_dir + '/' + fn
    if os.path.isfile(old_args):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}

        d = without_keys(json.loads(open(old_args, 'r').read()), ['to_train', 'epoch'])
        args.__dict__.update(d)

        """
        if overwrite_args:
            fn = args2fn(args)
        print(" New args:")
        print(args)
        """

        print('\nfilename: ', fn)
        mdl = model(args, fn)
        print(" [*] Loading model!")
        mdl.load(old_path)
        return mdl


def main():
    file_name = "lhc_e400_s1993_p0.0_h100_faffine_fl5_l1_dsdim16_dsl1_best"
    mdl = load_model(file_name)
    print(mdl)


if __name__ == "__main__":
    main()
