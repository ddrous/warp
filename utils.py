import os


def setup_run_folder(folder_path, script_name, datagen_folder=None):
    """ Copy the run script, the module files, and create a folder for the adaptation results. """

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Created a new run folder at:", folder_path)

    # Save the run scripts in that folder
    os.system(f"cp {script_name} {folder_path}")

    # Save the selfmod module files as well
    # module_folder = os.path.join(os.path.dirname(__file__), "../")
    module_folder = os.path.join(os.path.dirname(__file__))
    os.system(f"cp -r {module_folder} {folder_path}")
    print(" Backed up run script and module files ")

    ## Create a folder for the adaptation results
    adapt_folder = folder_path+"adapt/"
    if not os.path.exists(adapt_folder):
        os.mkdir(adapt_folder)
        print(" Created an adaptation folder at:", adapt_folder)

    ## Create a folder for the chckpoints results
    checkpoints_folder = folder_path+"checkpoints/"
    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)
        print(" Created a checkpoints folder at:", checkpoints_folder)

    ## Create a folder for the generation data scripts
    if datagen_folder is not None:
        data_folder = os.path.join(folder_path, "data/")
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
            print(" Created a data folder at:", data_folder)
        try:
            os.system(f"cp -r {datagen_folder}/ {data_folder}")
            print(" Attempting copy of the data generation scripts")
        except Exception as e:
            print(f" Could not copy the data generation scripts due to: {e}")
    else:
        data_folder = None

    return adapt_folder, checkpoints_folder, data_folder




def numpy_collate(batch):
  return jax.tree.map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
    
    self.num_batches = np.ceil(len(dataset) / batch_size).astype(int)
