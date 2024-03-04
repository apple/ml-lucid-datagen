#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import ast
import astunparse


class RewriteConfirm(ast.NodeTransformer):  # TODO: delete
    """Rewrites instances of `confirm` call.

    The rewrite addresses top-level occurrences of `confirm(x)` where `x` is any variable.
    This gets rewritten into `x.confirmed = True`. Any other form is ignored.
    """

    def visit_Module(self, node: ast.Module) -> ast.Module:
        match node:
            case ast.Module(body=[ast.Expr() as expr]):
                return ast.Module(body=[self._visit_top_level_expr(expr)])
        return node

    def _visit_top_level_expr(self, node: ast.Expr) -> ast.AST:
        match node:
            case ast.Expr(
                value=ast.Call(func=ast.Name(id="confirm"), args=[ast.Name(id=var_name)])
            ):
                return ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id=var_name), attr="confirmed")],
                    value=ast.Constant(value=True),
                )
        return node


class RewriteResume(ast.NodeTransformer):  # TODO: delete?
    """Rewrites instances of `confirm` call.

    The rewrite addresses top-level occurrences of `resume(x)` where `x` is any variable.
    This gets rewritten into `x`. Any other form is ignored.
    """

    def visit_Module(self, node: ast.Module) -> ast.Module:
        match node:
            case ast.Module(body=[ast.Expr() as expr]):
                return ast.Module(body=[self._visit_top_level_expr(expr)])
        return node

    def _visit_top_level_expr(self, node: ast.Expr) -> ast.AST:
        match node:
            case ast.Expr(
                value=ast.Call(func=ast.Name(id=func_name), args=[ast.Name(id=var_name)])
            ) if func_name == "resume":
                return ast.Expr(value=ast.Name(id=var_name))
        return node


def test_RewriteConfirm(input_: str, expected: str) -> None:
    ast_node = ast.parse(input_)
    rewritten_ast_node = RewriteConfirm().visit(ast_node)
    rewritten_code = astunparse.unparse(rewritten_ast_node).strip()
    assert expected == rewritten_code


def test_RewriteResume(input_: str, expected: str) -> None:
    ast_node = ast.parse(input_)
    rewritten_ast_node = RewriteResume().visit(ast_node)
    rewritten_code = astunparse.unparse(rewritten_ast_node).strip()
    assert expected == rewritten_code
