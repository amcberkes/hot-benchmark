"""Pinned Natural Earth base layers used in both climate maps."""
import hashlib
import subprocess

MAP_REV = "f1890d9f152c896d250a77557a5751a93d494776"
MAPS = {
    "ne_110m_admin_0_countries": "6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f",
    "ne_50m_admin_0_countries": "3e458fc036ad0a66411f2c1e6cac49c5d7bfb81cb1123bc513b22511a2b7fdeb",
    "ne_50m_admin_1_states_provinces_lakes": "b92c3f709b691240f6320e5cd7fade78cedfa3b91ecff14cecd3de0616c764b9",
    "ne_50m_lakes": "d350b75978b26fe839b797c2c529b2fb8f47fb3983c03f4964e36d5df9378a52",
}


def map_layers(directory, download):
    for layer, expected in MAPS.items():
        path = directory / (layer + ".geojson")
        if download and not path.exists():
            url = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{MAP_REV}/geojson/{path.name}"
            data = subprocess.run(
                ["curl", "-fLsS", "--retry", "2", "--max-time", "120", url],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Downloaded layer does not match {MAP_REV}: {path.name}")
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected Natural Earth layer: {path}")
