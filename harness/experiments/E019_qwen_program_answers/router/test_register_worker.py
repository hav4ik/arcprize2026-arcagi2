import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import register_worker as rw


def inst(iid, ip, web, ssh):
    return {"id": iid, "public_ipaddr": ip,
            "ports": {"8000/tcp": [{"HostPort": str(web)}], "22/tcp": [{"HostPort": str(ssh)}]}}


POOL = [inst(1, "1.1.1.1", 28572, 28326), inst(2, "1.1.1.1", 26960, 26717), inst(3, "2.2.2.2", 50300, 50347)]


class PickInstance(unittest.TestCase):
    def test_shared_ip_matches_by_web_port(self):
        self.assertEqual(rw.pick_instance(POOL, "1.1.1.1", 28572)["id"], 1)
        self.assertEqual(rw.pick_instance(POOL, "1.1.1.1", "26960")["id"], 2)

    def test_shared_ip_without_port_is_refused(self):
        with self.assertRaises(SystemExit):
            rw.pick_instance(POOL, "1.1.1.1")

    def test_unique_ip_without_port_ok(self):
        self.assertEqual(rw.pick_instance(POOL, "2.2.2.2")["id"], 3)

    def test_wrong_port_refused(self):
        with self.assertRaises(SystemExit):
            rw.pick_instance(POOL, "1.1.1.1", 1234)


if __name__ == "__main__":
    unittest.main()
