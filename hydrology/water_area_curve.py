from typing import Union, Optional

import numpy as np
import numpy.ma as ma
import rasterio

Number = Union[float, int]
ArrayLike = Union[Number, np.ndarray]


class HypsometricCurve:
    """
    Hypsometric curve builder and interpolators.
    Stores monotone arrays (z_sorted, S) built from a masked DEM.
    """

    def __init__(self,
                 dem: Optional[ma.MaskedArray] = None,
                 dem_path: Optional[str] = None,
                 pixel_area: Optional[float] = None) -> None:
        """
        Args:
            dem: masked DEM; masked cells are invalid.
            dem_path: path to a DEM raster; used if `dem` is None.
            pixel_area: constant pixel area in m^2; if reading from file and
                        None, it is inferred from affine transform.
        """
        if dem is None and dem_path is None:
            raise ValueError("Provide `dem` or `dem_path`")

        if dem is None:
            with rasterio.open(dem_path) as src:
                dem = src.read(1, masked=True)
                T = src.transform
                pixel_area = abs(T.a * T.e)

        if not isinstance(dem, ma.MaskedArray):
            raise TypeError("`dem` must be numpy.ma.MaskedArray")
        if pixel_area is None:
            raise ValueError("`pixel_area` required when `dem` is provided")

        z_sorted, S = self.build_area_level_table(dem, pixel_area)
        self.dem: ma.MaskedArray = dem
        self.z_sorted: np.ndarray = z_sorted
        self.S: np.ndarray = S
        self.pixel_area: float = float(pixel_area)

    def get_height_from_area(self,
                             target_area: ArrayLike) -> np.ndarray:
        """
        Interpolate height from area on stored (S -> z_sorted).
        """

        return np.interp(
            np.asarray(target_area, dtype="float64"),
            self.S,
            self.z_sorted,
            left=self.z_sorted[0],
            right=self.z_sorted[-1],
        )

    def get_area_from_height(self,
                             target_height: ArrayLike) -> np.ndarray:
        """
        Interpolate area from height on stored (z_sorted -> S).
        """
        return np.interp(
            np.asarray(target_height, dtype="float64"),
            self.z_sorted,
            self.S,
            left=0.0,
            right=self.S[-1],
        )

    def build_area_level_table(self,
                               dem: ma.MaskedArray,
                               pixel_area: float = 1.0,
                               ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build monotone arrays from a masked DEM.

        Returns:
            (z_sorted, S)
              z_sorted: ascending elevations [N]
              S: cumulative area [N], strictly increasing
        Raises:
            TypeError: if dem is not a masked array.
            ValueError: if no valid cells after masking.
        """
        if not isinstance(dem, ma.MaskedArray):
            raise TypeError("`dem` must be numpy.ma.MaskedArray")

        # merge NaNs into mask to avoid polluting sort
        if dem.dtype.kind == "f":
            nan_mask = np.isnan(dem.data)
            if nan_mask.any():
                dem = ma.masked_array(dem.data, mask=(dem.mask | nan_mask))

        z = dem.compressed().astype("float64")
        if z.size == 0:
            raise ValueError("DEM has no valid cells")

        z_sorted = np.sort(z)  # ascending
        N = z_sorted.size
        # assumes constant pixel area (projected CRS and uniform resolution)
        S = pixel_area * np.arange(1, N + 1, dtype="float64")  # strictly increasing
        return z_sorted, S

    def show(self):
        from matplotlib import pyplot as plt
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["axes.unicode_minus"] = False

        vmin, vmax = float(self.dem.min()), float(self.dem.max())
        N_sample = 50
        heights = np.linspace(vmin, vmax, num=N_sample)

        areas = self.get_area_from_height(heights)

        # plot
        fig_width, fig_height = (11, 8)
        font_size = round(fig_height * 2.25)
        plt.rcParams.update({'font.size': font_size})
        print(font_size)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        plt.plot(heights, areas, label="Hypsometric Curve", linewidth=font_size * 0.2, zorder=1)
        plt.scatter(heights, areas, color="red", s=font_size, marker="o", label="Sampled Area", zorder=2)
        plt.xlabel("Elevation (m)")
        plt.ylabel("Cumulative Area (m2)")
        plt.title("Hypsometric Curve of Lake")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
