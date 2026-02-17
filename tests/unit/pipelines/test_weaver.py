import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock

from src.pipelines.weaver import Weaver
from src.schemas.judge import JudgeDomain, JudgeResult, Operation, OperationType
from src.schemas.weaver import OpStatus
from src.storage.base import BaseVectorStore


class TestWeaver(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock Vector Store
        self.mock_store = Mock(spec=BaseVectorStore)
        self.mock_store.add = Mock(return_value=["new-id-123"])
        self.mock_store.update = Mock(return_value=True)
        self.mock_store.delete = Mock(return_value=True)

        # Mock Embedding Function
        self.mock_embed = Mock(return_value=[0.1, 0.2, 0.3])

        # Mock Graph Functions
        self.mock_create_event = AsyncMock(return_value="evt-new-123")
        self.mock_update_event = AsyncMock(return_value=True)
        self.mock_delete_event = AsyncMock(return_value=True)

        self.weaver = Weaver(
            vector_store=self.mock_store,
            embed_fn=self.mock_embed,
            graph_create_event=self.mock_create_event,
            graph_update_event=self.mock_update_event,
            graph_delete_event=self.mock_delete_event,
        )

    async def test_vector_add(self):
        op = Operation(type=OperationType.ADD, content="Test content")
        result = JudgeResult(operations=[op])
        
        res = await self.weaver.execute(result, JudgeDomain.PROFILE, "user1")
        
        self.assertEqual(res.succeeded, 1)
        self.assertEqual(res.executed[0].type, OperationType.ADD)
        self.assertEqual(res.executed[0].new_id, "new-id-123")
        
        self.mock_store.add.assert_called_once()
        call_args = self.mock_store.add.call_args
        self.assertEqual(call_args.kwargs['texts'], ["Test content"])
        self.assertEqual(call_args.kwargs['metadata'][0]['user_id'], "user1")

    async def test_vector_update_success(self):
        op = Operation(
            type=OperationType.UPDATE, 
            content="Updated content", 
            embedding_id="vec-001"
        )
        result = JudgeResult(operations=[op])
        
        res = await self.weaver.execute(result, JudgeDomain.SUMMARY, "user1")
        
        self.assertEqual(res.succeeded, 1)
        self.assertEqual(res.executed[0].type, OperationType.UPDATE)
        
        self.mock_store.update.assert_called_once()
        self.assertEqual(self.mock_store.update.call_args.kwargs['id'], "vec-001")

    async def test_vector_update_fallback(self):
        # Simulate UPDATE failing (ID not found)
        self.mock_store.update.return_value = False
        
        op = Operation(
            type=OperationType.UPDATE, 
            content="Updated content", 
            embedding_id="vec-missing"
        )
        result = JudgeResult(operations=[op])
        
        res = await self.weaver.execute(result, JudgeDomain.PROFILE, "user1")
        
        # Should report success because it fell back to ADD
        self.assertEqual(res.succeeded, 1)
        # But should show as UPDATE handled via fallback
        # Wait, the Weaver returns the result of _vector_add, which has type=ADD
        # Let's check the logic:
        # return await self._vector_add(op, domain, user_id)
        # _vector_add uses op.type which is UPDATE.
        # So it should report as UPDATE but with new_id set.
        
        self.assertEqual(res.executed[0].type, OperationType.UPDATE)
        self.assertEqual(res.executed[0].new_id, "new-id-123")
        
        self.mock_store.update.assert_called_once()
        self.mock_store.add.assert_called_once()  # Fallback triggered

    async def test_vector_delete(self):
        op = Operation(type=OperationType.DELETE, embedding_id="vec-001")
        result = JudgeResult(operations=[op])
        
        res = await self.weaver.execute(result, JudgeDomain.PROFILE, "user1")
        
        self.assertEqual(res.succeeded, 1)
        self.mock_store.delete.assert_called_once_with(ids=["vec-001"])

    async def test_graph_add(self):
        op = Operation(
            type=OperationType.ADD, 
            content="03-15 | Birthday | User's birthday"
        )
        result = JudgeResult(operations=[op])
        
        res = await self.weaver.execute(result, JudgeDomain.TEMPORAL, "user1")
        
        self.assertEqual(res.succeeded, 1)
        self.mock_create_event.assert_called_once()
        # Verify parsed args
        _, kwargs = self.mock_create_event.call_args
        self.assertEqual(kwargs['date_str'], "03-15")
        self.assertEqual(kwargs['event_data']['event_name'], "Birthday")

    async def test_graph_date_change_logic(self):
        # Simulate the DELETE + ADD pattern for date change
        ops = [
            Operation(type=OperationType.DELETE, embedding_id="evt-old"),
            Operation(
                type=OperationType.ADD, 
                content="02-10 | Dentist | New date"
            )
        ]
        result = JudgeResult(operations=ops)
        
        res = await self.weaver.execute(result, JudgeDomain.TEMPORAL, "user1")
        
        self.assertEqual(res.succeeded, 2)
        self.mock_delete_event.assert_called_once()
        self.mock_create_event.assert_called_once()

    async def test_guard_rails(self):
        # 1. Empty ADD
        op1 = Operation(type=OperationType.ADD, content="")
        # 2. UPDATE without ID
        op2 = Operation(type=OperationType.UPDATE, content="New stuff", embedding_id=None)
        
        result = JudgeResult(operations=[op1, op2])
        
        res = await self.weaver.execute(result, JudgeDomain.PROFILE, "user1")
        
        # Op 1 should be skipped
        self.assertEqual(res.executed[0].status, OpStatus.SKIPPED)
        
        # Op 2 should be converted to ADD and succeed
        self.assertEqual(res.executed[1].status, OpStatus.SUCCESS)
        self.assertEqual(res.executed[1].type, OperationType.ADD)  # Converted
        self.mock_store.add.assert_called_once()


if __name__ == "__main__":
    unittest.main()
