"""
    BBBC021 dataset class.

    Attributes:
        IMG_SHAPE : tuple
            Shape of the images: (C, H, W).
        CHANNELS : list
            Names of the channels.
        N_SITES : int
            Number of sites for each well.
        PLATES : list
            List of plate IDs.
        COMPOUNDS : list
            List of compounds.
        MOA : list
            List of Mechanisms of Action.
        index_vector : np.ndarray vector
            Vector of the absolute image indices of selected subset of BBBC021
            dataset 0...N where N is the number of images in the entirety of
            BBBC021 (including null MoA).
        image_df : pd.DataFrame
            Metadata for the subset of images that were selected in the form
            of a `DataFrame`.

    Methods:
        make_dataset(data_path=None)
            Creates a virtual HDF5 dataset with preprocessed images and
            metadata.
        download_raw_data(data_path=None)
            Downloads raw images and metadata.
    """

    IMG_SHAPE = constants.IMG_SHAPE
    CHANNELS = constants.CHANNELS
    N_SITES = constants.N_SITES
    PLATES = constants.PLATES
    COMPOUNDS = constants.COMPOUNDS
    MOA = constants.MOA

    def __init__(self, root_path="~/.cache/", **kwargs):
        """Initializes the BBBC021 dataset.

        Args:
            path : str, optional
                Path to the virtual HDF5 dataset.
                Default is '~/.cache/bbbc021/bbbc021.h5'.

        Returns: instance of the BBBC021 dataset
        """

        root_path = Path(root_path).expanduser()

        self.root_path = root_path

        self._paths = get_paths(root_path)

        compiled_hdf5_path = self._paths["compiled_hdf5"]

        if not compiled_hdf5_path.exists():
            raise RuntimeError(
                "Dataset not found at '{}'.\n Use BBBC021.download() to "
                "download raw data and BBBC021.make_dataset() to preprocess "
                "and create the dataset.".format(compiled_hdf5_path)
            )

        self.dataset = h5py.File(compiled_hdf5_path, "r")
        self.index_vector = np.arange(
            self.dataset["moa"].shape[0]  # pylint: disable=no-member
        )

        # filter dataset based on kwargs query

        for k, v in kwargs.items():
            if not isinstance(v, (list, tuple, set)):
                v = [v]

            bool_vector = np.zeros_like(self.index_vector)

            for e in v:
                if isinstance(e, str):
                    e = bytes(e, "utf-8")

                bool_vector = bool_vector + np.array(
                    self.dataset[k][self.index_vector] == e
                )

            self.index_vector = self.index_vector[np.flatnonzero(bool_vector)]

    @cached_property
    def image_df(self) -> pd.DataFrame:
        """
        Returns:
            Dataframe with columns:
                image_idx: the index of a given image in the original,
                    unfiltered BBBC021 dataset.
                relative_image_idx: assuming the BBBC021 dataset has been
                    filtered with selectors, the new index to access this image
                    as in:
                    ```python
                    bbbc021_subset = BBBC021(moa=['null'])
                    image, metadata = bbbc021_subset[relative_idx]
                    ```
        """

        return (
            pd.DataFrame(
                dict(
                    site=self.sites,
                    well=self.wells,
                    replicate=self.replicates,
                    plate=self.plates,
                    compound=self.compounds,
                    concentration=self.concentrations,
                    moa=self.moa,
                    image_idx=self.index_vector,
                    relative_image_idx=np.arange(len(self)),
                )
            )
        ).astype(
            dict(
                well="category",
                plate="category",
                compound="category",
                moa="category",
            )
        )

    @cached_property
    def moa_df(self) -> pd.DataFrame:
        """
        Return a 3 column `DataFrame` with every combination of compound,
        concentration, and mechanism-of-action.
        Includes compounds with unknown MoA.
        """
        return (
            self.image_df[["compound", "concentration", "moa"]]
            .drop_duplicates()
            .sort_values(["compound", "concentration"])
            .reset_index(drop=True)
        )

    def metadata(self, rel_index) -> Metadata:
        """
        Get metadata for the given image at `rel_index`.

        Args:
            rel_index: Relative image index 0...N where N is the number of
                images in the subset of the BBBC021 dataset you've selected
                when you created the `BBBC021` object.
        """

        row = self.image_df.iloc[rel_index]

        (
            site,
            well,
            replicate,
            plate,
            compound,
            concentration,
            moa,
            image_idx,
        ) = row[
            [
                "site",
                "well",
                "replicate",
                "plate",
                "compound",
                "concentration",
                "moa",
                "image_idx",
            ]
        ]

        metadata = Metadata(
            Plate(site, str(well), replicate, plate),
            Compound(compound, concentration, moa),
            image_idx,
        )

        return metadata

    def __getitem__(self, rel_index) -> Tuple[np.ndarray, Metadata]:

        abs_index = self.index_vector[rel_index]

        img = self.dataset["images"][abs_index].astype(np.float32)

        metadata = self.metadata(rel_index)

        return img, metadata

    def __len__(self):
        return len(self.index_vector)

    @property
    def images(self):
        """
        NOTE: This will load the entirety of the selected BBBC021 subset into
        memory.
        """
        return self.dataset["images"][self.index_vector]

    @cached_property
    def sites(self):
        return self.dataset["site"][self.index_vector]

    @cached_property
    def wells(self):
        return bytes_to_str(self.dataset["well"][self.index_vector])

    @cached_property
    def replicates(self):
        return self.dataset["replicate"][self.index_vector]

    @cached_property
    def plates(self):
        return bytes_to_str(self.dataset["plate"][self.index_vector])

    @cached_property
    def compounds(self):
        return bytes_to_str(self.dataset["compound"][self.index_vector])

    @cached_property
    def concentrations(self):
        return self.dataset["concentration"][self.index_vector]

    @cached_property
    def moa(self):
        return bytes_to_str(self.dataset["moa"][self.index_vector])
