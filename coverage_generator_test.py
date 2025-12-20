import unittest
import os
from app.coverage_generator import solve_with_astap
import pprint

class TestSolveWithASTAPReal(unittest.TestCase):
    def test_real_fits(self):
        astap_path = r"C:\Program Files\astap\astap_cli.exe"
        fits_path = r"C:\Users\kasat\Downloads\2024-11-30_19-57-11_-20_30_60_00s_0000.fits"

        self.assertTrue(os.path.exists(astap_path), f"ASTAP not found: {astap_path}")
        self.assertTrue(os.path.exists(fits_path), f"FITS file not found: {fits_path}")

        polygon = solve_with_astap(astap_path, fits_path)

        self.assertIsNotNone(polygon, "Solve failed or result TXT not found")
        self.assertEqual(len(polygon), 4, "Polygon should have 4 corners")

        print("Polygon corners:")
        pprint.pprint(polygon)

if __name__ == "__main__":
    unittest.main(buffer=False, verbosity=2)
