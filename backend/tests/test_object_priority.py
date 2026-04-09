"""Tests for COCO class priority map."""

from backend.services.object_priority import get_priority, DEFAULT_PRIORITY


class TestObjectPriority:
    def test_person_always_required(self):
        assert get_priority("person") == (True, 1.0)

    def test_car_stationary(self):
        assert get_priority("car", is_moving=False) == (False, 0.7)

    def test_car_moving(self):
        assert get_priority("car", is_moving=True) == (True, 0.7)

    def test_unknown_class_default(self):
        assert get_priority("unknown_class_xyz") == DEFAULT_PRIORITY

    def test_sports_ball_required(self):
        assert get_priority("sports ball") == (True, 0.8)

    def test_chair_not_required(self):
        flag, weight = get_priority("chair")
        assert flag is False
        assert weight == 0.3

    def test_dog_required(self):
        assert get_priority("dog") == (True, 0.9)

    def test_train_moving(self):
        assert get_priority("train", is_moving=True) == (True, 0.8)

    def test_train_stationary(self):
        assert get_priority("train", is_moving=False) == (False, 0.8)
