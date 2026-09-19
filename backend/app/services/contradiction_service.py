"""
Contradiction engine — detect conflicting inventory claims.

Checks:
- OUT quantity vs available stock
- Recent conflicting statements from same period

Returns: NO_CONFLICT | POSSIBLE_CONFLICT | STRONG_CONFLICT
Does NOT automatically reject all contradictions.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.statement import Statement
from app.models.baseline import Baseline
from app.models.enums import StatementStatus, DirectionEnum

RECENT_WINDOW_HOURS = 24


class ContradictionResult:
    def __init__(
        self,
        level: str,  # NO_CONFLICT | POSSIBLE_CONFLICT | STRONG_CONFLICT
        reason: str,
        available_stock: Optional[float] = None,
    ):
        self.level = level
        self.reason = reason
        self.available_stock = available_stock

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "reason": self.reason,
            "available_stock": self.available_stock,
        }


async def check_contradiction(
    db: AsyncSession,
    shop_id: uuid.UUID,
    product_id: Optional[uuid.UUID],
    quantity: float,
    direction: str,
) -> ContradictionResult:
    """
    Check for contradictions with current stock and recent statements.
    """
    if not product_id:
        return ContradictionResult("NO_CONFLICT", "No product resolved — skipping contradiction check.")

    dir_enum = DirectionEnum.IN if direction == "IN" else DirectionEnum.OUT

    # Calculate current available stock
    current_stock = await _calculate_stock(db, shop_id, product_id)

    # For OUT statements: check if we have enough stock
    if dir_enum == DirectionEnum.OUT and current_stock is not None:
        if quantity > current_stock:
            if current_stock <= 0:
                return ContradictionResult(
                    "STRONG_CONFLICT",
                    f"Cannot OUT {quantity} — current stock is {current_stock:.2f} (at or below zero).",
                    available_stock=current_stock,
                )
            elif quantity > current_stock * 1.5:
                return ContradictionResult(
                    "STRONG_CONFLICT",
                    f"OUT {quantity} exceeds current stock of {current_stock:.2f} by more than 50%.",
                    available_stock=current_stock,
                )
            else:
                return ContradictionResult(
                    "POSSIBLE_CONFLICT",
                    f"OUT {quantity} slightly exceeds current stock of {current_stock:.2f}.",
                    available_stock=current_stock,
                )

    # Check for conflicting recent statements
    recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)
    result = await db.execute(
        select(Statement).where(
            Statement.shop_id == shop_id,
            Statement.product_id == product_id,
            Statement.created_at >= recent_cutoff,
            Statement.status.in_([StatementStatus.CONFIRMED, StatementStatus.PENDING, StatementStatus.PROCESSING]),
        ).order_by(Statement.created_at.desc()).limit(10)
    )
    recent_statements = result.scalars().all()

    # Check for same-direction spike
    same_dir = [s for s in recent_statements if s.direction == dir_enum]
    if same_dir:
        recent_total = sum(float(s.quantity or 0) for s in same_dir)
        if recent_total > 0 and quantity > recent_total * 3:
            return ContradictionResult(
                "POSSIBLE_CONFLICT",
                f"New {direction} of {quantity} is much larger than recent {direction} statements (total {recent_total:.2f} in last {RECENT_WINDOW_HOURS}h).",
                available_stock=current_stock,
            )

    return ContradictionResult(
        "NO_CONFLICT",
        "No conflicts detected with recent statements or current stock.",
        available_stock=current_stock,
    )


async def _calculate_stock(
    db: AsyncSession,
    shop_id: uuid.UUID,
    product_id: uuid.UUID,
) -> Optional[float]:
    """Calculate current stock = latest baseline + confirmed IN - confirmed OUT."""
    # Get latest active baseline
    baseline_result = await db.execute(
        select(Baseline).where(
            Baseline.shop_id == shop_id,
            Baseline.product_id == product_id,
            Baseline.is_superseded == False,
        ).order_by(Baseline.created_at.desc()).limit(1)
    )
    baseline = baseline_result.scalar_one_or_none()

    baseline_qty = float(baseline.quantity) if baseline else 0.0
    baseline_time = baseline.created_at if baseline else None

    # Sum CONFIRMED IN after baseline (or all if no baseline)
    in_query = select(func.coalesce(func.sum(Statement.quantity), 0)).where(
        Statement.shop_id == shop_id,
        Statement.product_id == product_id,
        Statement.direction == DirectionEnum.IN,
        Statement.status == StatementStatus.CONFIRMED,
    )
    if baseline_time:
        in_query = in_query.where(Statement.created_at > baseline_time)
    in_result = await db.execute(in_query)
    in_total = float(in_result.scalar() or 0)

    # Sum CONFIRMED OUT after baseline (or all if no baseline)
    out_query = select(func.coalesce(func.sum(Statement.quantity), 0)).where(
        Statement.shop_id == shop_id,
        Statement.product_id == product_id,
        Statement.direction == DirectionEnum.OUT,
        Statement.status == StatementStatus.CONFIRMED,
    )
    if baseline_time:
        out_query = out_query.where(Statement.created_at > baseline_time)
    out_result = await db.execute(out_query)
    out_total = float(out_result.scalar() or 0)

    return baseline_qty + in_total - out_total
