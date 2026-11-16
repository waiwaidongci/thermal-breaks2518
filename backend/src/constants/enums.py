from enum import StrEnum
class Role(StrEnum):
 ADMIN='admin'
 ANALYST='analyst'
 VIEWER='viewer'
class RunStatus(StrEnum):
 QUEUED='queued'
 RUNNING='running'
 SUCCEEDED='succeeded'
 FAILED='failed'
class SegmentMode(StrEnum):
 FREE='free'
 MONOTONE_MEAN='monotone_mean'
class NoteStatus(StrEnum):
 DRAFT='draft'
 APPROVED='approved'
 REJECTED='rejected'
