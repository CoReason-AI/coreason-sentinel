from coreason_identity.models import UserContext


def test_user_context_initialization() -> None:
    context = UserContext(sub="user123", permissions=["admin", "editor"], extra="value")
    assert context.sub == "user123"
    assert context.permissions == ["admin", "editor"]
    assert context.extra == "value"


def test_user_context_defaults() -> None:
    context = UserContext()
    assert context.sub is None
    assert context.permissions is None
    assert context.missing_attr is None
