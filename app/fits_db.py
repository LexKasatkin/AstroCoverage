import sqlite3
import json

class FitsDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS fits_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                file_path TEXT UNIQUE,
                ra REAL, dec REAL, healpix INTEGER,
                min_ra REAL, max_ra REAL, min_dec REAL, max_dec REAL,
                fov_width REAL, fov_height REAL, pixel_scale REAL,
                hfd REAL, hfd_arcsec REAL, snr_median REAL, 
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
        self.conn.commit()

    def check_record_exists(self, file_path):
        self.cur.execute("SELECT 1 FROM fits_data WHERE file_path = ?", (file_path,))
        return self.cur.fetchone() is not None

    def insert_record(self, record):
        self.cur.execute("""
            INSERT OR REPLACE INTO fits_data (
                file_name, file_path, ra, dec, healpix,
                min_ra, max_ra, min_dec, max_dec,
                fov_width, fov_height, pixel_scale,
                hfd, hfd_arcsec, snr_median, fwhm_px, fwhm_arcsec, stars,
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
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record["file_name"], record["file_path"], record["ra"], record["dec"], record["healpix"],
            record["min_ra"], record["max_ra"], record["min_dec"], record["max_dec"],
            record["fov_width"], record["fov_height"], record["pixel_scale"],
            record["hfd"], record["hfd_arcsec"], record["snr_median"], record["fwhm_px"], record["fwhm_arcsec"], record["stars"],
            record["airmass"], record["altitude"], record["azimuth"],
            record["exptime"], record["gain"], record["offset"], record["ccd_temp"],
            record["camera"], record["telescope"], record["filter"],
            int(record["plate_solved"]), record["wcs_source"],
            record["date_obs"], record["date_loc"],
            record["latitude"], record["longitude"], record["elevation"],
            json.dumps(record["polygon"]), record["moon_alt"], record["moon_az"],
            record["sun_alt"], record["sun_az"], record.get("moon_phase"),
            record["sky_background"], record["sky_rms"], record["sky_brightness"],
            record["bortle"], record["bortle_float"]
        ))
        self.conn.commit()

        self.cur.execute("SELECT id FROM fits_data WHERE file_path = ?", (record["file_path"],))
        res = self.cur.fetchone()
        return res[0] if res else None

    def close(self):
        self.conn.close()
