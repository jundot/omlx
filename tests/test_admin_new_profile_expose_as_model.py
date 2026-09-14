"""Regression tests for API exposure in the new-profile form."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _model_settings_template() -> str:
    return (
        ROOT / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


def test_new_profile_api_toggle():
    html = _model_settings_template()
    form = html.split("<!-- New profile inline form (model scope) -->", 1)[1].split(
        "<!-- Edit profile inline dialog (model scope) -->", 1
    )[0]

    assert "modal.model_settings.profiles.expose_as_model" in form
    assert "newProfile.expose_as_model = !newProfile.expose_as_model" in form
    # The state labels are localised now, so assert the keys the toggle
    # renders rather than the English copy it used to hardcode.
    assert (
        "newProfile.expose_as_model ? "
        "t('modal.model_settings.profiles.expose_as_model_on') : "
        "t('modal.model_settings.profiles.expose_as_model_off')"
    ) in form


def test_new_profile_resets_api_exposure():
    html = _model_settings_template()
    opening = html.split("showNewProfileForm = true", 1)[1].split('"', 1)[0]

    assert "expose_as_model:false" in opening.replace(" ", "")


def test_create_profile_sends_api_exposure():
    script = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    body = script.split("async createProfile()", 1)[1].split(
        "async applyProfileToForm(", 1
    )[0]

    assert "expose_as_model: !!this.newProfile.expose_as_model" in body
