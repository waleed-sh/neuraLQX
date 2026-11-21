---
title: Profiling for scientists
---

# Profiling for scientists

These pages are for scientists who want to understand the cost of a simulation without working on the profiling implementation. You can begin with an ordinary [VMC run](../../concepts/drivers/running.md), enable recording, and inspect the operations that neuraLQX already annotates. Later, the same capture can include calculations from your own script.

A useful starting question is specific. Perhaps sampling takes longer after increasing an SU(2) cutoff, a new observable changes iteration time, or an SR calculation leaves the GPU idle between bursts of work. A profile helps connect that observation to the relevant part of the calculation.

## Begin with the built-in instrumentation

[Profiling a simulation](scientists/simulations.md) shows the minimal changes needed to record a driver run and locate its output. The supplied regions already identify the main phases, so you do not need to add decorators throughout your simulation script.

There are two independent controls to understand. The collection level selects application timings alone or application timings together with native runtime activity. The annotation detail selects how deeply neuraLQX names its own operations. Recording more package detail is different from tracing every Python call.

Start with a short, representative calculation. A whole production run can produce a large trace and make a local performance question harder to answer. [Choosing what to record](scientists/capture-options.md) explains the levels, capture schedules, synchronisation, and optional GPU counters.

## Add the parts that belong to your experiment

Your script may perform graph preparation, additional measurements, postprocessing, or a custom update outside the standard driver. [Profiling your own code](scientists/custom-code.md) shows how to give those operations names and include them in the same trace.

The distinction between Python and compiled work matters here. A scope around a Python call measures the time spent in that call. A name attached inside a compiled JAX function helps identify the generated operations. They answer different questions, and using both can connect a familiar scientific operation to its device execution.

## Read the result as an experiment

[Reading a profile](scientists/reading-results.md) explains the output files and the main timeline patterns. Look at repeated iterations, the location of waits, the relationship between host and device activity, and whether the recording is complete.

Compare calculations under controlled conditions. Graph topology, representation cutoff, sample count, discarded samples, network size, operator construction, chunking, precision, and hardware can all affect the result. Record those choices alongside the profile. A faster calculation with a different estimator or poorer sampling is not automatically an improvement for the scientific question.

Compilation also needs its own interpretation. To study startup, include it. To study repeated iterations, first execute the same workload with matching shapes and dtypes, then measure a later interval. Warming the native tracing machinery is a separate operation from compiling the calculation.

## What you do not need to decide yet

The initial examples use one process and the normal output layout. You can follow them without choosing a distributed collection transport, understanding clock calibration, or changing the event buffer limits. Those topics become relevant when a capture spans several processes or grows large enough to lose events.

After the practical sequence, the [instrumentation guide](instrumentation.md) explains the measurement boundaries in more detail. The final [configuration reference](configuration.md) provides the complete list of profiling controls when you need a particular default or combination.
