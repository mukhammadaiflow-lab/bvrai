"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight, MoreHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button, ButtonProps } from "./button";

interface PaginationProps extends React.ComponentPropsWithoutRef<"nav"> {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  siblingCount?: number;
  showFirstLast?: boolean;
  size?: "sm" | "default" | "lg";
}

const Pagination = ({
  className,
  currentPage,
  totalPages,
  onPageChange,
  siblingCount = 1,
  showFirstLast = true,
  size = "default",
  ...props
}: PaginationProps) => {
  const range = React.useMemo(() => {
    const totalPageNumbers = siblingCount * 2 + 3;

    if (totalPages <= totalPageNumbers) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
    const rightSiblingIndex = Math.min(currentPage + siblingCount, totalPages);

    const shouldShowLeftDots = leftSiblingIndex > 2;
    const shouldShowRightDots = rightSiblingIndex < totalPages - 1;

    if (!shouldShowLeftDots && shouldShowRightDots) {
      const leftItemCount = 3 + 2 * siblingCount;
      const leftRange = Array.from({ length: leftItemCount }, (_, i) => i + 1);
      return [...leftRange, "dots", totalPages];
    }

    if (shouldShowLeftDots && !shouldShowRightDots) {
      const rightItemCount = 3 + 2 * siblingCount;
      const rightRange = Array.from(
        { length: rightItemCount },
        (_, i) => totalPages - rightItemCount + i + 1
      );
      return [1, "dots", ...rightRange];
    }

    if (shouldShowLeftDots && shouldShowRightDots) {
      const middleRange = Array.from(
        { length: rightSiblingIndex - leftSiblingIndex + 1 },
        (_, i) => leftSiblingIndex + i
      );
      return [1, "dots", ...middleRange, "dots", totalPages];
    }

    return [];
  }, [currentPage, totalPages, siblingCount]);

  const buttonSize = size === "sm" ? "icon-sm" : "icon";

  return (
    <nav
      role="navigation"
      aria-label="pagination"
      className={cn("mx-auto flex w-full justify-center", className)}
      {...props}
    >
      <ul className="flex flex-row items-center gap-1">
        {showFirstLast && (
          <li>
            <PaginationButton
              size={buttonSize}
              onClick={() => onPageChange(1)}
              disabled={currentPage === 1}
              aria-label="Go to first page"
            >
              <ChevronLeft className="h-4 w-4" />
              <ChevronLeft className="h-4 w-4 -ml-2" />
            </PaginationButton>
          </li>
        )}
        <li>
          <PaginationButton
            size={buttonSize}
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
            aria-label="Go to previous page"
          >
            <ChevronLeft className="h-4 w-4" />
          </PaginationButton>
        </li>
        {range.map((pageNumber, index) =>
          pageNumber === "dots" ? (
            <li key={`dots-${index}`}>
              <span className="flex h-10 w-10 items-center justify-center">
                <MoreHorizontal className="h-4 w-4" />
              </span>
            </li>
          ) : (
            <li key={pageNumber}>
              <PaginationButton
                size={buttonSize}
                variant={currentPage === pageNumber ? "default" : "outline"}
                onClick={() => onPageChange(pageNumber as number)}
                aria-label={`Go to page ${pageNumber}`}
                aria-current={currentPage === pageNumber ? "page" : undefined}
              >
                {pageNumber}
              </PaginationButton>
            </li>
          )
        )}
        <li>
          <PaginationButton
            size={buttonSize}
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            aria-label="Go to next page"
          >
            <ChevronRight className="h-4 w-4" />
          </PaginationButton>
        </li>
        {showFirstLast && (
          <li>
            <PaginationButton
              size={buttonSize}
              onClick={() => onPageChange(totalPages)}
              disabled={currentPage === totalPages}
              aria-label="Go to last page"
            >
              <ChevronRight className="h-4 w-4" />
              <ChevronRight className="h-4 w-4 -ml-2" />
            </PaginationButton>
          </li>
        )}
      </ul>
    </nav>
  );
};
Pagination.displayName = "Pagination";

const PaginationButton = React.forwardRef<
  HTMLButtonElement,
  ButtonProps
>(({ className, variant = "outline", ...props }, ref) => (
  <Button
    ref={ref}
    variant={variant}
    className={cn("", className)}
    {...props}
  />
));
PaginationButton.displayName = "PaginationButton";

interface PaginationInfoProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  className?: string;
}

const PaginationInfo = ({
  currentPage,
  totalPages,
  totalItems,
  itemsPerPage,
  className,
}: PaginationInfoProps) => {
  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);

  return (
    <p className={cn("text-sm text-muted-foreground", className)}>
      Showing <span className="font-medium">{startItem}</span> to{" "}
      <span className="font-medium">{endItem}</span> of{" "}
      <span className="font-medium">{totalItems}</span> results
    </p>
  );
};
PaginationInfo.displayName = "PaginationInfo";

interface SimplePaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  onPageChange: (page: number) => void;
  className?: string;
}

const SimplePagination = ({
  currentPage,
  totalPages,
  totalItems,
  itemsPerPage,
  onPageChange,
  className,
}: SimplePaginationProps) => {
  if (totalPages <= 1) return null;

  return (
    <div
      className={cn(
        "flex flex-col sm:flex-row items-center justify-between gap-4 py-4",
        className
      )}
    >
      <PaginationInfo
        currentPage={currentPage}
        totalPages={totalPages}
        totalItems={totalItems}
        itemsPerPage={itemsPerPage}
      />
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={onPageChange}
      />
    </div>
  );
};
SimplePagination.displayName = "SimplePagination";

export {
  Pagination,
  PaginationButton,
  PaginationInfo,
  SimplePagination,
};
