from fastapi.testclient import TestClient

from core.api.server import app

c = TestClient(app)


def test_health():
    assert c.get("/health").json() == {"ok": True}


def test_moderate_endpoint():
    r = c.post(
        "/moderate", json={"text": "Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?"}
    )
    assert r.status_code == 200
    assert r.json()["action"] in ("WARN", "BLOCK", "MASK")
