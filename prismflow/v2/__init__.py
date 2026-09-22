"""PrismFlow V2 -- semantic multi-angle analysis.

V2 reuses V1's thesis (agreement counts in proportion to effective
independence, not headcount) but replaces synthetic views with real,
externally-sourced evidence: one data source per angle.

V1 (``prismflow/models``, ``prismflow/statistics``, ``prismflow/eniv``,
``prismflow/evaluation``) is FROZEN at tag ``v1-final``. Nothing under
``prismflow/v2`` may modify it; suspected V1 defects are reported, not
patched. See ``docs/CONTRACT.md`` section 6.
"""
