# ============================================================
# engine/brain.py
# ============================================================

from dataclasses import dataclass

import pandas as pd
import ace_lib as ace


@dataclass
class BrainContext:
    """
    Shared WorldQuant BRAIN context.

    This class does not implement or duplicate BRAIN API behavior.
    It delegates catalog access to the existing ace_lib functions.
    """

    session: object
    region: str
    universe: str
    dataset_id: str
    delay: int = 1

    # --------------------------------------------------------
    # Dataset catalog
    # --------------------------------------------------------

    def get_datasets(self) -> pd.DataFrame:
        """
        Retrieve the live BRAIN dataset catalog for the
        configured region/universe.
        """

        return ace.get_datasets(
            self.session,
            region=self.region,
            universe=self.universe,
            delay=self.delay,
        )

    # --------------------------------------------------------
    # Datafield catalog
    # --------------------------------------------------------

    def get_datafields(self) -> pd.DataFrame:
        """
        Retrieve the live BRAIN datafield catalog for the
        configured region/universe.
        """

        return ace.get_datafields(
            self.session,
            region=self.region,
            universe=self.universe,
            delay=self.delay,
        )

    # --------------------------------------------------------
    # Operator catalog
    # --------------------------------------------------------

    def get_operators(self) -> pd.DataFrame:
        """
        Retrieve the live BRAIN operator catalog.
        """

        return ace.get_operators(
            self.session
        )

    # --------------------------------------------------------
    # Fields belonging to this dataset
    # --------------------------------------------------------

    def get_dataset_fields(self) -> pd.DataFrame:
        """
        Return only fields belonging to self.dataset_id.

        The filtering is performed locally against the live
        datafield catalog returned by BRAIN.
        """

        fields = self.get_datafields().copy()

        if "dataset_id" not in fields.columns:
            raise KeyError(
                "BRAIN datafield catalog does not contain "
                "'dataset_id'."
            )

        dataset_ids = (
            fields["dataset_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        filtered = fields[
            dataset_ids.eq(
                str(self.dataset_id).strip()
            )
        ].copy()

        return (
            filtered
            .reset_index(drop=True)
        )