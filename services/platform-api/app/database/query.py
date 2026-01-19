"""
Advanced Query Builder

Fluent SQL query builder with:
- Type-safe query construction
- SQL injection prevention
- Query composition and reuse
- Subquery support
- CTEs (Common Table Expressions)
- Window functions
- JSON operations
"""

from typing import (
    Optional, Dict, Any, List, Union, Tuple, TypeVar, Generic,
    Callable, Type, Sequence,
)
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from abc import ABC, abstractmethod
import re
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Operator(str, Enum):
    """SQL operators."""
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT EXISTS"
    ANY = "ANY"
    ALL = "ALL"
    CONTAINS = "@>"
    CONTAINED_BY = "<@"
    OVERLAP = "&&"


class JoinType(str, Enum):
    """SQL join types."""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL OUTER JOIN"
    CROSS = "CROSS JOIN"
    LATERAL = "LATERAL JOIN"


class OrderDirection(str, Enum):
    """Order direction."""
    ASC = "ASC"
    DESC = "DESC"


class AggregateFunction(str, Enum):
    """SQL aggregate functions."""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    ARRAY_AGG = "ARRAY_AGG"
    STRING_AGG = "STRING_AGG"
    JSON_AGG = "JSON_AGG"
    JSONB_AGG = "JSONB_AGG"


@dataclass
class QueryResult:
    """Result of a query execution."""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    query: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return self.row_count

    def first(self) -> Optional[Dict[str, Any]]:
        """Get the first row."""
        return self.rows[0] if self.rows else None

    def scalar(self, column: str = None) -> Any:
        """Get a single scalar value."""
        if not self.rows:
            return None
        row = self.rows[0]
        if column:
            return row.get(column)
        return list(row.values())[0] if row else None

    def column(self, name: str) -> List[Any]:
        """Get all values from a column."""
        return [row.get(name) for row in self.rows]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rows": self.rows,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
        }


class Expression:
    """SQL expression builder."""

    def __init__(self, expression: str, params: Optional[Dict[str, Any]] = None):
        self.expression = expression
        self.params = params or {}

    def __str__(self) -> str:
        return self.expression

    def __and__(self, other: "Expression") -> "Expression":
        return Expression(
            f"({self.expression} AND {other.expression})",
            {**self.params, **other.params},
        )

    def __or__(self, other: "Expression") -> "Expression":
        return Expression(
            f"({self.expression} OR {other.expression})",
            {**self.params, **other.params},
        )

    def __invert__(self) -> "Expression":
        return Expression(f"NOT ({self.expression})", self.params)

    @classmethod
    def raw(cls, sql: str, **params) -> "Expression":
        """Create a raw SQL expression."""
        return cls(sql, params)

    @classmethod
    def column(cls, name: str, table: Optional[str] = None) -> "Expression":
        """Create a column reference."""
        if table:
            return cls(f'"{table}"."{name}"')
        return cls(f'"{name}"')

    @classmethod
    def literal(cls, value: Any, param_name: str = None) -> "Expression":
        """Create a literal value."""
        if param_name is None:
            param_name = f"lit_{id(value)}"
        return cls(f":{param_name}", {param_name: value})

    @classmethod
    def null(cls) -> "Expression":
        """Create NULL expression."""
        return cls("NULL")

    @classmethod
    def case(
        cls,
        cases: List[Tuple["Expression", "Expression"]],
        else_value: Optional["Expression"] = None,
    ) -> "Expression":
        """Create a CASE expression."""
        parts = ["CASE"]
        params = {}

        for condition, result in cases:
            parts.append(f"WHEN {condition.expression} THEN {result.expression}")
            params.update(condition.params)
            params.update(result.params)

        if else_value:
            parts.append(f"ELSE {else_value.expression}")
            params.update(else_value.params)

        parts.append("END")
        return cls(" ".join(parts), params)

    @classmethod
    def coalesce(cls, *expressions: "Expression") -> "Expression":
        """Create a COALESCE expression."""
        exprs = ", ".join(e.expression for e in expressions)
        params = {}
        for e in expressions:
            params.update(e.params)
        return cls(f"COALESCE({exprs})", params)

    @classmethod
    def cast(cls, expr: "Expression", type_name: str) -> "Expression":
        """Create a CAST expression."""
        return cls(f"CAST({expr.expression} AS {type_name})", expr.params)


class Condition:
    """Condition builder for WHERE clauses."""

    def __init__(self):
        self._conditions: List[Expression] = []
        self._operator = "AND"

    def where(
        self,
        column: str,
        operator: Union[Operator, str],
        value: Any = None,
        param_name: str = None,
    ) -> "Condition":
        """Add a condition."""
        op = operator.value if isinstance(operator, Operator) else operator

        if param_name is None:
            param_name = f"p_{len(self._conditions)}"

        if op in ("IS NULL", "IS NOT NULL"):
            self._conditions.append(Expression(f'"{column}" {op}'))
        elif op == "IN" or op == "NOT IN":
            placeholders = ", ".join(f":{param_name}_{i}" for i in range(len(value)))
            params = {f"{param_name}_{i}": v for i, v in enumerate(value)}
            self._conditions.append(Expression(f'"{column}" {op} ({placeholders})', params))
        elif op == "BETWEEN":
            self._conditions.append(Expression(
                f'"{column}" BETWEEN :{param_name}_low AND :{param_name}_high',
                {f"{param_name}_low": value[0], f"{param_name}_high": value[1]},
            ))
        else:
            self._conditions.append(Expression(f'"{column}" {op} :{param_name}', {param_name: value}))

        return self

    def eq(self, column: str, value: Any) -> "Condition":
        """Add equality condition."""
        return self.where(column, Operator.EQ, value)

    def ne(self, column: str, value: Any) -> "Condition":
        """Add not equal condition."""
        return self.where(column, Operator.NE, value)

    def lt(self, column: str, value: Any) -> "Condition":
        """Add less than condition."""
        return self.where(column, Operator.LT, value)

    def le(self, column: str, value: Any) -> "Condition":
        """Add less than or equal condition."""
        return self.where(column, Operator.LE, value)

    def gt(self, column: str, value: Any) -> "Condition":
        """Add greater than condition."""
        return self.where(column, Operator.GT, value)

    def ge(self, column: str, value: Any) -> "Condition":
        """Add greater than or equal condition."""
        return self.where(column, Operator.GE, value)

    def like(self, column: str, pattern: str) -> "Condition":
        """Add LIKE condition."""
        return self.where(column, Operator.LIKE, pattern)

    def ilike(self, column: str, pattern: str) -> "Condition":
        """Add case-insensitive LIKE condition."""
        return self.where(column, Operator.ILIKE, pattern)

    def in_(self, column: str, values: List[Any]) -> "Condition":
        """Add IN condition."""
        return self.where(column, Operator.IN, values)

    def not_in(self, column: str, values: List[Any]) -> "Condition":
        """Add NOT IN condition."""
        return self.where(column, Operator.NOT_IN, values)

    def is_null(self, column: str) -> "Condition":
        """Add IS NULL condition."""
        return self.where(column, Operator.IS_NULL)

    def is_not_null(self, column: str) -> "Condition":
        """Add IS NOT NULL condition."""
        return self.where(column, Operator.IS_NOT_NULL)

    def between(self, column: str, low: Any, high: Any) -> "Condition":
        """Add BETWEEN condition."""
        return self.where(column, Operator.BETWEEN, (low, high))

    def raw(self, expression: str, **params) -> "Condition":
        """Add a raw condition."""
        self._conditions.append(Expression(expression, params))
        return self

    def and_(self) -> "Condition":
        """Switch to AND mode."""
        self._operator = "AND"
        return self

    def or_(self) -> "Condition":
        """Switch to OR mode."""
        self._operator = "OR"
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the condition clause."""
        if not self._conditions:
            return "", {}

        clause = f" {self._operator} ".join(c.expression for c in self._conditions)
        params = {}
        for c in self._conditions:
            params.update(c.params)

        return clause, params


class QueryBuilder(ABC):
    """Base query builder."""

    def __init__(self):
        self._params: Dict[str, Any] = {}
        self._param_counter = 0

    def _next_param(self, prefix: str = "p") -> str:
        """Generate next parameter name."""
        self._param_counter += 1
        return f"{prefix}_{self._param_counter}"

    @abstractmethod
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the SQL query and parameters."""
        pass

    def to_sql(self) -> str:
        """Get the SQL string only."""
        sql, _ = self.build()
        return sql

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters only."""
        _, params = self.build()
        return params


class SelectQuery(QueryBuilder):
    """
    SELECT query builder.

    Usage:
        query = (SelectQuery()
            .select("id", "name", "email")
            .from_table("users")
            .where("status", "=", "active")
            .where("created_at", ">", date(2024, 1, 1))
            .order_by("created_at", "DESC")
            .limit(10)
            .offset(0))

        sql, params = query.build()
    """

    def __init__(self):
        super().__init__()
        self._distinct = False
        self._columns: List[str] = []
        self._from_clause: Optional[str] = None
        self._joins: List[Tuple[JoinType, str, str]] = []
        self._conditions = Condition()
        self._group_by: List[str] = []
        self._having = Condition()
        self._order_by: List[Tuple[str, OrderDirection]] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._for_update = False
        self._skip_locked = False
        self._ctes: List[Tuple[str, "SelectQuery"]] = []

    def distinct(self) -> "SelectQuery":
        """Add DISTINCT clause."""
        self._distinct = True
        return self

    def select(self, *columns: str) -> "SelectQuery":
        """Add columns to select."""
        self._columns.extend(columns)
        return self

    def select_all(self) -> "SelectQuery":
        """Select all columns."""
        self._columns = ["*"]
        return self

    def select_raw(self, expression: str) -> "SelectQuery":
        """Add a raw expression to select."""
        self._columns.append(expression)
        return self

    def count(self, column: str = "*", alias: str = "count") -> "SelectQuery":
        """Add COUNT aggregate."""
        self._columns.append(f"COUNT({column}) AS {alias}")
        return self

    def sum(self, column: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add SUM aggregate."""
        alias = alias or f"sum_{column}"
        self._columns.append(f"SUM({column}) AS {alias}")
        return self

    def avg(self, column: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add AVG aggregate."""
        alias = alias or f"avg_{column}"
        self._columns.append(f"AVG({column}) AS {alias}")
        return self

    def max(self, column: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add MAX aggregate."""
        alias = alias or f"max_{column}"
        self._columns.append(f"MAX({column}) AS {alias}")
        return self

    def min(self, column: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add MIN aggregate."""
        alias = alias or f"min_{column}"
        self._columns.append(f"MIN({column}) AS {alias}")
        return self

    def from_table(self, table: str, alias: Optional[str] = None) -> "SelectQuery":
        """Set the FROM clause."""
        if alias:
            self._from_clause = f'"{table}" AS "{alias}"'
        else:
            self._from_clause = f'"{table}"'
        return self

    def from_subquery(self, subquery: "SelectQuery", alias: str) -> "SelectQuery":
        """Set FROM to a subquery."""
        sql, params = subquery.build()
        self._from_clause = f"({sql}) AS {alias}"
        self._params.update(params)
        return self

    def join(
        self,
        table: str,
        condition: str,
        join_type: JoinType = JoinType.INNER,
        alias: Optional[str] = None,
    ) -> "SelectQuery":
        """Add a JOIN clause."""
        table_ref = f'"{table}"'
        if alias:
            table_ref = f'"{table}" AS "{alias}"'
        self._joins.append((join_type, table_ref, condition))
        return self

    def left_join(self, table: str, condition: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add a LEFT JOIN."""
        return self.join(table, condition, JoinType.LEFT, alias)

    def right_join(self, table: str, condition: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add a RIGHT JOIN."""
        return self.join(table, condition, JoinType.RIGHT, alias)

    def inner_join(self, table: str, condition: str, alias: Optional[str] = None) -> "SelectQuery":
        """Add an INNER JOIN."""
        return self.join(table, condition, JoinType.INNER, alias)

    def where(
        self,
        column: str,
        operator: Union[Operator, str],
        value: Any = None,
    ) -> "SelectQuery":
        """Add a WHERE condition."""
        self._conditions.where(column, operator, value)
        return self

    def where_eq(self, column: str, value: Any) -> "SelectQuery":
        """Add equality WHERE condition."""
        self._conditions.eq(column, value)
        return self

    def where_in(self, column: str, values: List[Any]) -> "SelectQuery":
        """Add WHERE IN condition."""
        self._conditions.in_(column, values)
        return self

    def where_null(self, column: str) -> "SelectQuery":
        """Add WHERE IS NULL condition."""
        self._conditions.is_null(column)
        return self

    def where_not_null(self, column: str) -> "SelectQuery":
        """Add WHERE IS NOT NULL condition."""
        self._conditions.is_not_null(column)
        return self

    def where_between(self, column: str, low: Any, high: Any) -> "SelectQuery":
        """Add WHERE BETWEEN condition."""
        self._conditions.between(column, low, high)
        return self

    def where_raw(self, expression: str, **params) -> "SelectQuery":
        """Add a raw WHERE condition."""
        self._conditions.raw(expression, **params)
        return self

    def or_where(
        self,
        column: str,
        operator: Union[Operator, str],
        value: Any = None,
    ) -> "SelectQuery":
        """Add an OR WHERE condition."""
        self._conditions.or_().where(column, operator, value)
        return self

    def group_by(self, *columns: str) -> "SelectQuery":
        """Add GROUP BY clause."""
        self._group_by.extend(columns)
        return self

    def having(
        self,
        column: str,
        operator: Union[Operator, str],
        value: Any = None,
    ) -> "SelectQuery":
        """Add HAVING condition."""
        self._having.where(column, operator, value)
        return self

    def order_by(
        self,
        column: str,
        direction: Union[OrderDirection, str] = OrderDirection.ASC,
    ) -> "SelectQuery":
        """Add ORDER BY clause."""
        if isinstance(direction, str):
            direction = OrderDirection(direction.upper())
        self._order_by.append((column, direction))
        return self

    def order_by_desc(self, column: str) -> "SelectQuery":
        """Add ORDER BY DESC."""
        return self.order_by(column, OrderDirection.DESC)

    def limit(self, count: int) -> "SelectQuery":
        """Set LIMIT clause."""
        self._limit = count
        return self

    def offset(self, count: int) -> "SelectQuery":
        """Set OFFSET clause."""
        self._offset = count
        return self

    def paginate(self, page: int, per_page: int) -> "SelectQuery":
        """Set pagination."""
        self._limit = per_page
        self._offset = (page - 1) * per_page
        return self

    def for_update(self, skip_locked: bool = False) -> "SelectQuery":
        """Add FOR UPDATE clause."""
        self._for_update = True
        self._skip_locked = skip_locked
        return self

    def with_cte(self, name: str, query: "SelectQuery") -> "SelectQuery":
        """Add a CTE (Common Table Expression)."""
        self._ctes.append((name, query))
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the SQL query."""
        parts = []
        params = dict(self._params)

        # CTEs
        if self._ctes:
            cte_parts = []
            for name, query in self._ctes:
                cte_sql, cte_params = query.build()
                cte_parts.append(f"{name} AS ({cte_sql})")
                params.update(cte_params)
            parts.append("WITH " + ", ".join(cte_parts))

        # SELECT
        select_clause = "SELECT"
        if self._distinct:
            select_clause += " DISTINCT"

        columns = ", ".join(self._columns) if self._columns else "*"
        parts.append(f"{select_clause} {columns}")

        # FROM
        if self._from_clause:
            parts.append(f"FROM {self._from_clause}")

        # JOINs
        for join_type, table, condition in self._joins:
            parts.append(f"{join_type.value} {table} ON {condition}")

        # WHERE
        where_clause, where_params = self._conditions.build()
        if where_clause:
            parts.append(f"WHERE {where_clause}")
            params.update(where_params)

        # GROUP BY
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        # HAVING
        having_clause, having_params = self._having.build()
        if having_clause:
            parts.append(f"HAVING {having_clause}")
            params.update(having_params)

        # ORDER BY
        if self._order_by:
            order_parts = [f"{col} {dir.value}" for col, dir in self._order_by]
            parts.append(f"ORDER BY {', '.join(order_parts)}")

        # LIMIT
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")

        # OFFSET
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        # FOR UPDATE
        if self._for_update:
            clause = "FOR UPDATE"
            if self._skip_locked:
                clause += " SKIP LOCKED"
            parts.append(clause)

        return " ".join(parts), params


class InsertQuery(QueryBuilder):
    """
    INSERT query builder.

    Usage:
        query = (InsertQuery("users")
            .columns("name", "email", "status")
            .values("John Doe", "john@example.com", "active")
            .returning("id"))

        sql, params = query.build()
    """

    def __init__(self, table: str):
        super().__init__()
        self._table = table
        self._columns: List[str] = []
        self._values: List[List[Any]] = []
        self._on_conflict: Optional[str] = None
        self._conflict_columns: List[str] = []
        self._update_on_conflict: Dict[str, Any] = {}
        self._returning: List[str] = []
        self._from_select: Optional[SelectQuery] = None

    def columns(self, *columns: str) -> "InsertQuery":
        """Set columns to insert."""
        self._columns = list(columns)
        return self

    def values(self, *values: Any) -> "InsertQuery":
        """Add a row of values."""
        self._values.append(list(values))
        return self

    def values_dict(self, data: Dict[str, Any]) -> "InsertQuery":
        """Add values from a dictionary."""
        if not self._columns:
            self._columns = list(data.keys())
        self._values.append([data.get(col) for col in self._columns])
        return self

    def values_dicts(self, data_list: List[Dict[str, Any]]) -> "InsertQuery":
        """Add multiple rows from dictionaries."""
        for data in data_list:
            self.values_dict(data)
        return self

    def from_select(self, query: SelectQuery) -> "InsertQuery":
        """Insert from a SELECT query."""
        self._from_select = query
        return self

    def on_conflict_do_nothing(self, *columns: str) -> "InsertQuery":
        """Add ON CONFLICT DO NOTHING."""
        self._on_conflict = "DO NOTHING"
        self._conflict_columns = list(columns)
        return self

    def on_conflict_do_update(
        self,
        conflict_columns: List[str],
        update_columns: Dict[str, Any],
    ) -> "InsertQuery":
        """Add ON CONFLICT DO UPDATE (upsert)."""
        self._on_conflict = "DO UPDATE"
        self._conflict_columns = conflict_columns
        self._update_on_conflict = update_columns
        return self

    def returning(self, *columns: str) -> "InsertQuery":
        """Add RETURNING clause."""
        self._returning = list(columns)
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the SQL query."""
        params = {}

        # INSERT INTO
        columns_str = ", ".join(f'"{c}"' for c in self._columns)
        parts = [f'INSERT INTO "{self._table}" ({columns_str})']

        # VALUES or SELECT
        if self._from_select:
            select_sql, select_params = self._from_select.build()
            parts.append(select_sql)
            params.update(select_params)
        else:
            value_rows = []
            for i, row in enumerate(self._values):
                row_params = []
                for j, value in enumerate(row):
                    param_name = f"v_{i}_{j}"
                    row_params.append(f":{param_name}")
                    params[param_name] = value
                value_rows.append(f"({', '.join(row_params)})")
            parts.append(f"VALUES {', '.join(value_rows)}")

        # ON CONFLICT
        if self._on_conflict:
            conflict_cols = ", ".join(f'"{c}"' for c in self._conflict_columns)
            if self._on_conflict == "DO NOTHING":
                parts.append(f"ON CONFLICT ({conflict_cols}) DO NOTHING")
            elif self._on_conflict == "DO UPDATE":
                update_parts = []
                for col, value in self._update_on_conflict.items():
                    param_name = f"upd_{col}"
                    update_parts.append(f'"{col}" = :{param_name}')
                    params[param_name] = value
                parts.append(f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {', '.join(update_parts)}")

        # RETURNING
        if self._returning:
            parts.append(f"RETURNING {', '.join(self._returning)}")

        return " ".join(parts), params


class UpdateQuery(QueryBuilder):
    """
    UPDATE query builder.

    Usage:
        query = (UpdateQuery("users")
            .set("status", "inactive")
            .set("updated_at", datetime.utcnow())
            .where("id", "=", user_id)
            .returning("id", "status"))

        sql, params = query.build()
    """

    def __init__(self, table: str):
        super().__init__()
        self._table = table
        self._sets: Dict[str, Any] = {}
        self._conditions = Condition()
        self._returning: List[str] = []

    def set(self, column: str, value: Any) -> "UpdateQuery":
        """Set a column value."""
        self._sets[column] = value
        return self

    def set_dict(self, data: Dict[str, Any]) -> "UpdateQuery":
        """Set multiple columns from a dictionary."""
        self._sets.update(data)
        return self

    def set_raw(self, column: str, expression: str) -> "UpdateQuery":
        """Set a column to a raw expression."""
        self._sets[column] = Expression(expression)
        return self

    def increment(self, column: str, amount: int = 1) -> "UpdateQuery":
        """Increment a column."""
        self._sets[column] = Expression(f'"{column}" + {amount}')
        return self

    def decrement(self, column: str, amount: int = 1) -> "UpdateQuery":
        """Decrement a column."""
        self._sets[column] = Expression(f'"{column}" - {amount}')
        return self

    def where(
        self,
        column: str,
        operator: Union[Operator, str],
        value: Any = None,
    ) -> "UpdateQuery":
        """Add a WHERE condition."""
        self._conditions.where(column, operator, value)
        return self

    def where_eq(self, column: str, value: Any) -> "UpdateQuery":
        """Add equality WHERE condition."""
        self._conditions.eq(column, value)
        return self

    def returning(self, *columns: str) -> "UpdateQuery":
        """Add RETURNING clause."""
        self._returning = list(columns)
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the SQL query."""
        params = {}

        # UPDATE
        parts = [f'UPDATE "{self._table}" SET']

        # SET
        set_parts = []
        for col, value in self._sets.items():
            if isinstance(value, Expression):
                set_parts.append(f'"{col}" = {value.expression}')
                params.update(value.params)
            else:
                param_name = f"set_{col}"
                set_parts.append(f'"{col}" = :{param_name}')
                params[param_name] = value

        parts.append(", ".join(set_parts))

        # WHERE
        where_clause, where_params = self._conditions.build()
        if where_clause:
            parts.append(f"WHERE {where_clause}")
            params.update(where_params)

        # RETURNING
        if self._returning:
            parts.append(f"RETURNING {', '.join(self._returning)}")

        return " ".join(parts), params


class DeleteQuery(QueryBuilder):
    """
    DELETE query builder.

    Usage:
        query = (DeleteQuery("users")
            .where("status", "=", "deleted")
            .where("deleted_at", "<", cutoff_date)
            .returning("id"))

        sql, params = query.build()
    """

    def __init__(self, table: str):
        super().__init__()
        self._table = table
        self._conditions = Condition()
        self._returning: List[str] = []

    def where(
        self,
        column: str,
        operator: Union[Operator, str],
        value: Any = None,
    ) -> "DeleteQuery":
        """Add a WHERE condition."""
        self._conditions.where(column, operator, value)
        return self

    def where_eq(self, column: str, value: Any) -> "DeleteQuery":
        """Add equality WHERE condition."""
        self._conditions.eq(column, value)
        return self

    def where_in(self, column: str, values: List[Any]) -> "DeleteQuery":
        """Add WHERE IN condition."""
        self._conditions.in_(column, values)
        return self

    def returning(self, *columns: str) -> "DeleteQuery":
        """Add RETURNING clause."""
        self._returning = list(columns)
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the SQL query."""
        params = {}

        # DELETE FROM
        parts = [f'DELETE FROM "{self._table}"']

        # WHERE
        where_clause, where_params = self._conditions.build()
        if where_clause:
            parts.append(f"WHERE {where_clause}")
            params.update(where_params)

        # RETURNING
        if self._returning:
            parts.append(f"RETURNING {', '.join(self._returning)}")

        return " ".join(parts), params


class RawQuery(QueryBuilder):
    """
    Raw SQL query with parameter binding.

    Usage:
        query = RawQuery(
            "SELECT * FROM users WHERE status = :status AND created_at > :date",
            status="active",
            date=date(2024, 1, 1),
        )

        sql, params = query.build()
    """

    def __init__(self, sql: str, **params):
        super().__init__()
        self._sql = sql
        self._params = params

    def bind(self, **params) -> "RawQuery":
        """Add parameter bindings."""
        self._params.update(params)
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the SQL query."""
        return self._sql, self._params
