"""Network slicing simulator for the safety-floor self-sufficiency study."""
from .slices import Slice, default_slices
from .traffic import TrafficModel, SCENARIOS
from .environment import SlicingEnv
from .allocators import all_allocators, EqualAllocator, DemandProportionalAllocator, PrioritySafeAllocator
from .metrics import summarise, print_report
