import h5py

def inspect_hdf5_file(filename='new_chetan_eyes_close_epochs.h5'):
    """
    Print the structure of the HDF5 file created by the raw‐epoch extraction script.
    """
    try:
        with h5py.File(filename, 'r') as h5f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"📂 Group: {name}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"📄 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

            print(f"\n🔍 Inspecting HDF5 File: {filename}\n")
            h5f.visititems(print_structure)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filename}")

if __name__ == "__main__":
    inspect_hdf5_file()
