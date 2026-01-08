import sqlite3
import json


class FitsDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS fits_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT,
                    file_path TEXT UNIQUE,
                    ra REAL, dec REAL, healpix INTEGER,
                    min_ra REAL, max_ra REAL, min_dec REAL, max_dec REAL,
                    fov_width REAL, fov_height REAL, pixel_scale REAL,
                    hfd REAL, hfd_arcsec REAL, snr_median REAL, sky_snr REAL,
                    fwhm_px REAL, fwhm_arcsec REAL, stars INTEGER,
                    airmass REAL, altitude REAL, azimuth REAL,
                    exptime REAL, gain REAL, offset REAL, ccd_temp REAL,
                    camera TEXT, telescope TEXT, filter TEXT,
                    plate_solved INTEGER, wcs_source TEXT,
                    date_obs TEXT, date_loc TEXT,
                    latitude REAL, longitude REAL, elevation REAL,
                    polygon TEXT, moon_alt REAL, moon_az REAL,
                    sun_alt REAL, sun_az REAL, moon_phase REAL,
                    sky_background REAL, sky_rms REAL, sky_brightness REAL,
                    bortle INTEGER, bortle_float REAL
                )
            """)

    def check_record_exists(self, file_path: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM fits_data WHERE file_path = ?",
            (file_path,)
        )
        return cur.fetchone() is not None

    def insert_many(self, records: list[dict]):
        if not records:
            return

        sql = """
            INSERT OR REPLACE INTO fits_data (
                file_name, file_path, ra, dec, healpix,
                min_ra, max_ra, min_dec, max_dec,
                fov_width, fov_height, pixel_scale,
                hfd, hfd_arcsec, snr_median, sky_snr,
                fwhm_px, fwhm_arcsec, stars,
                airmass, altitude, azimuth,
                exptime, gain, offset, ccd_temp,
                camera, telescope, filter,
                plate_solved, wcs_source,
                date_obs, date_loc,
                latitude, longitude, elevation,
                polygon, moon_alt, moon_az,
                sun_alt, sun_az, moon_phase,
                sky_background, sky_rms, sky_brightness,
                bortle, bortle_float
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """

        values = [
            (
                r["file_name"], r["file_path"], r["ra"], r["dec"], r["healpix"],
                r["min_ra"], r["max_ra"], r["min_dec"], r["max_dec"],
                r["fov_width"], r["fov_height"], r["pixel_scale"],
                r["hfd"], r["hfd_arcsec"], r["snr_median"], r["sky_snr"],
                r["fwhm_px"], r["fwhm_arcsec"], r["stars"],
                r["airmass"], r["altitude"], r["azimuth"],
                r["exptime"], r["gain"], r["offset"], r["ccd_temp"],
                r["camera"], r["telescope"], r["filter"],
                int(r["plate_solved"]), r["wcs_source"],
                r["date_obs"], r["date_loc"],
                r["latitude"], r["longitude"], r["elevation"],
                json.dumps(r["polygon"]),
                r["moon_alt"], r["moon_az"],
                r["sun_alt"], r["sun_az"], r.get("moon_phase"),
                r["sky_background"], r["sky_rms"], r["sky_brightness"],
                r["bortle"], r["bortle_float"]
            )
            for r in records
        ]

        with self.conn:
            self.conn.executemany(sql, values)

    def close(self):
        self.conn.close()
