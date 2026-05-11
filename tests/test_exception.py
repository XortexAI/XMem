import pytest

from src.utils.exceptions import (
    XMemError,
    ConfigurationError,
    ValidationError,
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreValidationError,
    VectorNotFoundError,
    DatabaseError,
    DatabaseConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMContextLengthError,
    EmbeddingError,
)


class TestXMemError:

    def test_str_includes_operation_when_provided(self):
        err = XMemError("something broke", operation="save")
        assert str(err) == "[save] something broke"

    def test_str_returns_message_only_when_no_operation(self):
        err = XMemError("something broke")
        assert str(err) == "something broke"

    def test_mssg_attribute_is_accessible(self):
        err = XMemError("hello")
        assert str(err) == "hello"

    def test_operation_is_none_by_default(self):
        err = XMemError("message")
        assert err.operation == None

    def test_operation_attribute_is_accessible(self):
        err = XMemError("msg", operation="delete")
        assert err.operation == "delete"

    def test_details_defaults_to_empty_dict_when_not_provided(self):
        err = XMemError("msg")
        assert err.details == {}

    def test_details_defaults_to_empty_dict_when_none_passed(self):
        err = XMemError("msg", details=None)
        assert err.details == {}

    def test_details_stores_provided_dict(self):
        err = XMemError("msg", details={"user_id": "123", "count": 5})
        assert err.details["user_id"] == "123"
        assert err.details["count"] == 5

    def test_repr_contains_class_name(self):
        err = XMemError("msg", operation="op")
        assert "XMemError" in repr(err)

    def test_repr_contains_message_and_operation(self):
        err = XMemError("msg", operation="op")
        assert "msg" in repr(err)
        assert "op" in repr(err)


class TestToDict:

    def test_to_dict_has_required_keys(self):
        err = XMemError("msg")
        result = err.to_dict()
        assert "error" in result
        assert "message" in result
        assert "operation" in result
        assert "details" in result

    def test_to_dict_error_key_is_class_name(self):
        err = XMemError("msg")
        assert err.to_dict()["error"] == "XMemError"

    def test_to_dict_subclass_uses_subclass_name(self):
        err = ValidationError("bad input")
        assert err.to_dict()["error"] == "ValidationError"

    def test_to_dict_message_matches(self):
        err = XMemError("something went wrong", operation="fetch")
        assert err.to_dict()["message"] == "something went wrong"

    def test_to_dict_operation_matches(self):
        err = XMemError("msg", operation="fetch")
        assert err.to_dict()["operation"] == "fetch"

    def test_to_dict_details_matches(self):
        err = XMemError("msg", details={"key": "value"})
        assert err.to_dict()["details"] == {"key": "value"}


class TestExceptionHierarchy:

    def test_configuration_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise ConfigurationError("missing key")

    def test_validation_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise ValidationError("bad input")

    def test_vector_store_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise VectorStoreError("store failed")

    def test_vector_store_connection_error_is_caught_as_vector_store_error(
            self):
        with pytest.raises(VectorStoreError):
            raise VectorStoreConnectionError("cannot connect")

    def test_vector_store_connection_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise VectorStoreConnectionError("cannot connect")

    def test_vector_not_found_error_is_caught_as_vector_store_error(self):
        with pytest.raises(VectorStoreError):
            raise VectorNotFoundError("id not found")

    def test_database_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise DatabaseError("db failed")

    def test_database_connection_error_is_caught_as_database_error(self):
        with pytest.raises(DatabaseError):
            raise DatabaseConnectionError("cannot connect to mongo")

    def test_llm_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise LLMError("llm failed")

    def test_llm_rate_limit_error_is_caught_as_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMRateLimitError("rate limited")

    def test_llm_context_length_error_is_caught_as_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMContextLengthError("too long")

    def test_embedding_error_is_caught_as_xmem_error(self):
        with pytest.raises(XMemError):
            raise EmbeddingError("embed failed")

    def test_validation_error_is_not_a_vector_store_error(self):
        err = ValidationError("bad input")
        assert not isinstance(err, VectorStoreError)
