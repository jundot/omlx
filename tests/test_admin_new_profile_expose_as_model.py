"""#3648 — the New Profile form could name a profile's API name but not expose it.

Assigning an api name at creation implies the profile is API-addressable. It was
not: `Expose as model` lived only in the edit dialog, so the profile had to be
saved and reopened with the pencil icon to become reachable. The create endpoint
has always accepted `expose_as_model` (`CreateProfileRequest`), so the gap was in
the form alone.

Same shape as the other admin-UI regression tests in this directory: read the
template and the script as text and pin the wiring, because there is no browser
here to click.
"""

from pathlib import Path


def _model_settings_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


def _dashboard_script() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "omlx/admin/static/js/dashboard.js").read_text()


def _new_profile_form(html: str) -> str:
    return html.split("<!-- New profile inline form (model scope) -->", 1)[1].split(
        "<!-- Edit profile inline dialog (model scope) -->", 1
    )[0]


def test_the_new_profile_form_offers_the_api_toggle():
    form = _new_profile_form(_model_settings_template())

    assert "newProfile.expose_as_model" in form
    assert "modal.model_settings.profiles.expose_as_model" in form


def test_the_toggle_reads_and_writes_the_same_flag():
    # A button that only reads, or only writes, looks right and does nothing.
    form = _new_profile_form(_model_settings_template())

    assert "newProfile.expose_as_model = !newProfile.expose_as_model" in form
    assert "newProfile.expose_as_model ? 'ON' : 'OFF'" in form


def test_the_form_resets_the_flag_when_it_opens():
    # Without this the flag survives from a previous open, so a second profile
    # silently inherits the first one's choice.
    html = _model_settings_template()

    assert "showNewProfileForm = true" in html
    opening = html.split("showNewProfileForm = true", 1)[1][:400]
    assert "expose_as_model:false" in opening.replace(" ", "")


def test_create_sends_the_flag_to_the_endpoint():
    # The half the form cannot do on its own: `createProfile()` builds the POST
    # body by hand, and a field it does not name is a field the backend never
    # sees, however the toggle is drawn.
    script = _dashboard_script()
    body = script.split("async createProfile()", 1)[1].split("async ", 1)[0]

    assert "expose_as_model" in body
    assert "this.newProfile.expose_as_model" in body


def test_the_default_is_still_off():
    # The accept control. Exposing a profile on the API is not a neutral
    # default: a profile created without asking must not become addressable.
    html = _model_settings_template()
    script = _dashboard_script()
    opening = html.split("showNewProfileForm = true", 1)[1][:400]

    assert "expose_as_model:false" in opening.replace(" ", "")
    body = script.split("async createProfile()", 1)[1].split("async ", 1)[0]
    assert "!!this.newProfile.expose_as_model" in body


def test_the_edit_dialog_still_has_its_own_toggle():
    # The second accept control: adding one to the create form must not have
    # moved the one that was already working.
    html = _model_settings_template()
    edit = html.split("<!-- Edit profile inline dialog (model scope) -->", 1)[1]

    assert "_editExposeAsModel" in edit
