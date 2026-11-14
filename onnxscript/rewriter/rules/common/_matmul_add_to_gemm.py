# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Does the following transformation:
- Add(MatMul(X, W), B) -> Gemm
- Add(MatMul(Transpose(X), W), B) -> Gemm
- Add(MatMul(X, Transpose(W)), B) -> Gemm
- Add(MatMul(Transpose(X), Transpose(W)), B) -> Gemm
"""

import abc
from typing import ClassVar

import onnx_ir as ir

from onnxscript.rewriter import _ir_utils
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


def _is_valid_2d_transpose(trans_out: ir.Value | None) -> tuple[bool, str]:
    """Validate that the Transpose node transposes a 2D matrix (perm=[1, 0] or omitted)."""
    if trans_out is None:
        return True, ""
    node = trans_out.producer()
    if node is None or node.op_type != "Transpose":
        return True, ""
    perm_attr = node.attributes.get("perm")
    if perm_attr is None:
        # In ONNX Transpose, if perm is omitted, dimensions are reversed.
        # For rank 2 tensors, reversing (0, 1) gives [1, 0].
        return True, ""
    if hasattr(perm_attr, "is_ref") and perm_attr.is_ref():
        return (
            False,
            f"Transpose 'perm' refers to reference attribute {perm_attr.ref_attr_name!r}",
        )
    try:
        perm_vals = list(perm_attr.as_ints())
    except (TypeError, ValueError, AttributeError):
        return False, "Transpose 'perm' attribute could not be read as integers"
    if perm_vals != [1, 0]:
        return False, f"Expected Transpose perm [1, 0], got {perm_vals}"
    return True, ""


class _MatMulAddToGemmBase(RewriteRuleClassBase, abc.ABC):
    trans_a: ClassVar = False
    trans_b: ClassVar = False

    def rewrite(self, op, input_a, input_b, input_c, **_):
        attributes = {}
        if self.trans_a:
            attributes["transA"] = 1
        if self.trans_b:
            attributes["transB"] = 1
        return op.Gemm(input_a, input_b, input_c, **attributes)

    def check(
        self,
        context,
        input_a,
        input_b,
        input_c,
        trans_a_out=None,
        trans_b_out=None,
        **_,
    ):
        del context  # Not used
        check_result = MatchResult()
        # Rank of input_a and input_b must be 2
        if not (_ir_utils.has_rank(input_a, 2) and _ir_utils.has_rank(input_b, 2)):
            return check_result.fail("Rank of input_a and input_b must be 2")
        valid_a, reason_a = _is_valid_2d_transpose(trans_a_out)
        if not valid_a:
            return check_result.fail(reason_a)
        valid_b, reason_b = _is_valid_2d_transpose(trans_b_out)
        if not valid_b:
            return check_result.fail(reason_b)
        # Gemm only broadcasts C to (M, N), while Add can also grow the output,
        # e.g. a (1, K) x (K, N) MatMul plus a (2, N) bias gives (2, N).
        m = _ir_utils.get_dim(input_a, 1 if self.trans_a else 0)
        n = _ir_utils.get_dim(input_b, 0 if self.trans_b else 1)
        c_shape = input_c.shape
        if c_shape is None or c_shape.rank() > 2:
            return check_result.fail("input_c must have a known shape of rank <= 2")
        for c_dim, out_dim in zip(reversed(c_shape.dims), (n, m)):
            if c_dim != 1 and not _ir_utils.same_dim(c_dim, out_dim):
                return check_result.fail(
                    f"input_c shape {c_shape} does not broadcast to ({m}, {n})"
                )
        return check_result


class MatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(a, b), c)`` with ``Gemm(a, b, c)``."""

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(input_a, input_b)
        return op.Add(matmul, input_c)


class TransAMatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(Transpose(a), b), c)`` with ``Gemm(a, b, c)``."""

    trans_a: ClassVar = True

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(
            op.Transpose(input_a, _outputs=["trans_a_out"]),
            input_b,
        )
        return op.Add(matmul, input_c)


class TransBMatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(a, Transpose(b)), c)`` with ``Gemm(a, b, c)``."""

    trans_b: ClassVar = True

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(
            input_a,
            op.Transpose(input_b, _outputs=["trans_b_out"]),
        )
        return op.Add(matmul, input_c)


class TransABMatMulAddToGemm(_MatMulAddToGemmBase):
    """Replaces ``Add(MatMul(Transpose(a), Transpose(b)), c)`` with ``Gemm(a, b, c)``."""

    trans_a: ClassVar = True
    trans_b: ClassVar = True

    def pattern(self, op, input_a, input_b, input_c):
        matmul = op.MatMul(
            op.Transpose(input_a, perm=[1, 0]),
            op.Transpose(input_b, perm=[1, 0]),
        )
        return op.Add(matmul, input_c)


matmul_add_to_gemm_rule = MatMulAddToGemm().rule()
transpose_a_matmul_add_to_gemm_rule = TransAMatMulAddToGemm().rule()
transpose_b_matmul_add_to_gemm_rule = TransBMatMulAddToGemm().rule()
transpose_ab_matmul_add_to_gemm_rule = TransABMatMulAddToGemm().rule()


rules = RewriteRuleSet(
    [
        transpose_ab_matmul_add_to_gemm_rule,
        transpose_a_matmul_add_to_gemm_rule,
        transpose_b_matmul_add_to_gemm_rule,
        matmul_add_to_gemm_rule,
    ]
)
