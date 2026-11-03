# -*- coding: utf-8 -*-
"""Проверки данных: критические и информационные пункты."""
from __future__ import annotations

from src.loader_adapt import load_split
from src.validate_adapt import full_report, notes, problems


def test_no_critical_problems():
    train, test = load_split()
    report = full_report(train, test)
    assert problems(report) == []


def test_report_covers_key_points():
    train, test = load_split()
    names = [r["проверка"] for r in full_report(train, test)]
    for expected in (
        "Число признаков",
        "Служебные колонки не попали в признаки",
        "Эксперименты Train и Test не пересекаются",
        "Пропуски в значениях сенсоров",
    ):
        assert expected in names


def test_constant_features_are_informational_only():
    train, test = load_split()
    report = full_report(train, test)
    const = [r for r in report if r["проверка"] == "Постоянные признаки"][0]
    assert const["уровень"] == "информация"
    assert const["результат"] in {"OK", "ВНИМАНИЕ"}
    if const["результат"] == "ВНИМАНИЕ":
        assert const in notes(report)
        assert const not in problems(report)


def test_every_check_has_level():
    train, test = load_split()
    for item in full_report(train, test):
        assert item["уровень"] in {"критично", "информация"}
        assert item["результат"] in {"OK", "ВНИМАНИЕ", "ПРОБЛЕМА"}
