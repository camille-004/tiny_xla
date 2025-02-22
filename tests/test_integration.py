from tiny_xla.core.graph import Graph
from tiny_xla.core.operation import Operation, OpType
from tiny_xla.core.types import Tensor, XLAType
from tiny_xla.ops import (
    add,
    batch_norm,
    constant,
    conv2d,
    max_pool2d,
    multiply,
    parameter,
    placeholder,
    relu,
)
from tiny_xla.passes.memory_analysis import create_memory_analysis_pass
from tiny_xla.passes.optimizations.dead_code_elimination import create_dce_pass
from tiny_xla.passes.pass_manager import (
    OptimizationLevel,
    PassManager,
    PassManagerConfig,
)


def test_simple_math():
    """Test compilation pipeline with simple math operations."""
    x = placeholder((32, 128), XLAType.FLOAT32, "input_x")
    y = placeholder((32, 128), XLAType.FLOAT32, "input_y")

    c1 = constant([2.0], (), XLAType.FLOAT32)
    c2 = constant([3.0], (), XLAType.FLOAT32)

    t1 = multiply(x, c1)
    t2 = multiply(y, c2)
    output = add(t1, t2)

    graph = Graph()
    output_id = graph.add_op(output)

    pm = PassManager(
        PassManagerConfig(
            optimization_level=OptimizationLevel.NORMAL, verify_each_pass=True
        )
    )

    pm.add_pass(create_dce_pass())
    pm.add_pass(create_memory_analysis_pass())

    result = pm.run(graph, outputs=[output_id])
    assert result.successful_passes > 0

    assert graph.validate()

    for op in graph.ops.values():
        assert hasattr(op.metadata, "memory_offset")
        assert hasattr(op.metadata, "memory_size")


def test_conv_net_pipeline():
    """Test compilation pipeline with CNN opertaions."""
    input_shape = (1, 224, 224, 3)

    images = placeholder(input_shape, XLAType.FLOAT32, "input_images")

    conv1_kernel = parameter((3, 3, 3, 64), XLAType.FLOAT32, "conv1_kernel")
    conv1 = conv2d(
        images,
        conv1_kernel,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        channels_out=64,
    )

    scale = parameter((64,), XLAType.FLOAT32, "bn1_scale")
    offset = parameter((64,), XLAType.FLOAT32, "bn1_offset")
    mean = parameter((64,), XLAType.FLOAT32, "bn1_mean")
    var = parameter((64,), XLAType.FLOAT32, "bn1_var")

    bn1 = batch_norm(conv1, scale, offset, mean, var)

    relu1 = relu(bn1)

    pool1 = max_pool2d(
        relu1, pool_size=(2, 2), strides=(2, 2), padding="valid"
    )

    graph = Graph()
    output_id = graph.add_op(pool1)

    pm = PassManager(
        PassManagerConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            verify_each_pass=True,
        )
    )

    pm.add_pass(create_dce_pass(), group_name="cleanup")
    pm.add_pass(create_memory_analysis_pass(), group_name="memory")

    result = pm.run(graph, outputs=[output_id])
    assert result.successful_passes > 0

    assert graph.validate()

    assert output_id in graph.ops

    memory_stats = result.pass_stats["memory_analysis"]

    assert memory_stats is not None
    assert memory_stats.total_memory > 0
    assert memory_stats.peak_memory > 0


def test_dead_code_elimination():
    """Test dead code elimination in the pipeline."""
    x = placeholder((10,), XLAType.FLOAT32, "input")
    y = constant([1.0], (), XLAType.FLOAT32)

    t1 = add(x, y)

    dead1 = multiply(x, y)
    dead2 = add(dead1, y)

    graph = Graph()
    live_id = graph.add_op(t1)
    dead_ops = [graph.add_op(dead1), graph.add_op(dead2)]

    initial_size = len(graph.ops)

    pm = PassManager()
    pm.add_pass(create_dce_pass())
    result = pm.run(graph, outputs=[live_id])  # noqa:F841

    assert len(graph.ops) < initial_size
    assert live_id in graph.ops
    assert all(op_id not in graph.ops for op_id in dead_ops)


def test_memory_analysis_with_reuse():
    """Test memory analysis with buffer reuse opportunities."""
    input_data = placeholder((1000,), XLAType.FLOAT32, "input")

    t1 = add(input_data, constant([1.0], (), XLAType.FLOAT32))
    t2 = multiply(t1, constant([2.0], (), XLAType.FLOAT32))
    t3 = add(t2, constant([3.0], (), XLAType.FLOAT32))

    graph = Graph()
    output_id = graph.add_op(t3)

    pm = PassManager()
    pm.add_pass(create_memory_analysis_pass())
    result = pm.run(graph, outputs=[output_id])

    memory_stats = result.pass_stats["memory_analysis"]

    assert memory_stats is not None
    assert memory_stats.reused_buffers > 0
    assert memory_stats.peak_memory < memory_stats.total_memory * 0.8


def test_shape_error_handling():
    """Test handling of shape validation errors."""
    x = placeholder((10, 20), XLAType.FLOAT32, "input")
    y = placeholder((10, 20), XLAType.FLOAT32, "input")
    result = add(x, y)

    graph = Graph()
    output_id = graph.add_op(result)

    class BadShapeOp(Operation):
        def __init__(self, orig_op: Operation):
            super().__init__(OpType.ELEMENT_WISE, "bad_shape", [])
            self._orig_op = orig_op

        def compute_output_type(self) -> Tensor:
            return Tensor(XLAType.FLOAT32, (-1, -1))

        def validate(self) -> bool:
            return True

        def lower(self) -> Operation:
            return self

    graph.ops[output_id] = BadShapeOp(graph.ops[output_id])

    pm = PassManager()
    pm.add_pass(create_memory_analysis_pass())

    stats = pm.run(graph, outputs=[output_id])
    assert stats.failed_passes > 0
    assert "memory_analysis" not in stats.pass_stats
