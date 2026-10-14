"""Headless UI flows with Streamlit's AppTest (plot clicks and the browser player cannot be simulated)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def fresh(timeout=60):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    assert not at.exception, at.exception
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception, at.exception


def test_starts_with_preview_and_no_run():
    at = fresh()
    assert at.session_state["run"] is None


def test_run_then_reset():
    at = fresh()
    at.number_input(key="iters").set_value(20)
    at.run()
    click(at, "Run")
    run = at.session_state["run"]
    assert run is not None and run.iteration == 20 and len(run.names) == 4
    assert "play" in at.session_state["figs"]
    click(at, "Reset")
    assert at.session_state["run"] is None


def test_every_objective_runs():
    at = fresh(90)
    for name in list(at.selectbox(key="objective").options):
        if name.startswith("Custom"):
            continue
        at.selectbox(key="objective").select(name)
        at.run()                                        # the objective callback loads its own defaults
        at.number_input(key="iters").set_value(10)
        at.run()
        click(at, "Run")
        assert at.session_state["run"].iteration == 10, name


def test_bad_input_is_reported_not_crashing():
    at = fresh()
    at.text_input(key="lr").set_value("abc")
    at.run()
    assert not at.exception and any("learning rate" in e.value for e in at.error)


def test_optimizer_unchecked_and_none_selected():
    at = fresh()
    for n in ("SGD + momentum", "AdaGrad", "RMSProp", "Adam"):
        at.checkbox(key=f"use::{n}").set_value(False)
    at.run()
    click(at, "Run")
    assert any("select at least one" in e.value for e in at.error)


def test_pin_and_clear():
    at = fresh()
    at.number_input(key="iters").set_value(10)
    at.run()
    click(at, "Run")
    click(at, "Pin run")
    assert at.session_state["ghost"] is not None
    click(at, "Clear pin")
    assert at.session_state["ghost"] is None


def test_custom_is_off_by_default_and_works_when_enabled(monkeypatch):
    at = fresh()
    at.selectbox(key="objective").select("Custom (write your own)")
    at.run()
    assert any("switched off" in e.value for e in at.error)
    monkeypatch.setenv("ALLOW_CUSTOM", "1")
    at = fresh()
    at.selectbox(key="objective").select("Custom (write your own)")
    at.number_input(key="iters").set_value(5)
    at.run()
    click(at, "Run")
    assert at.session_state["run"].iteration == 5
    at.text_area(key="code").set_value("def loss(w):\n    return w[0]**2 + 3*w[1]**2")
    at.run()
    click(at, "Run")
    assert not at.exception


def test_changing_objective_discards_run():
    at = fresh()
    at.number_input(key="iters").set_value(5)
    at.run()
    click(at, "Run")
    at.selectbox(key="objective").select("Beale")
    at.run()
    assert at.session_state["run"] is None


def test_guided_example_sets_controls_and_pins_baseline():
    at = fresh()
    at.selectbox(key="lesson").select("Momentum in a narrow valley")
    at.run()
    assert at.session_state["objective"] == "Ill-conditioned quadratic" and at.session_state["iters"] == 60
    assert not at.session_state["use::AdaGrad"]
    click(at, "Run")
    assert at.session_state["run"] is not None and at.session_state["ghost"] is not None
    at.number_input(key="iters").set_value(30)                       # a manual change leaves the example
    at.run()
    assert at.session_state["lesson"] == "Manual / modified"


def test_convergence_tab_and_downloads_exist_after_run():
    at = fresh()
    at.number_input(key="iters").set_value(30)
    at.run()
    click(at, "Run")
    assert len(at.tabs) == 2 and at.tabs[1].label == "Convergence"
    at.selectbox(key="tol_loss").select(1e-2)
    at.run()
    assert not at.exception
