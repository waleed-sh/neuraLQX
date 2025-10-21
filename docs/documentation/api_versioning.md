# API Versioning and Stability

Introduction
------------

This document explains how the version numbering scheme of neuraLQX works (e.g., **A.B.C**), what 
guarantees we provide (and do *not* provide) about API stability, and how you, as a user of the 
library, should interpret and rely on version numbers when writing your code.

### Why versioning matters

Version numbers are not mere bookkeeping: they signal *compatibility guarantees*, *feature 
availability*, *bug-fix policy*, and *API change policy*. By understanding them, you can make correct
choices about when to upgrade, when to pin a version, and how to structure your code to avoid surprises 
when updating.

Version Numbering Scheme (A.B.C)
--------------------------------

We adopt the semantic style versioning scheme (often called "Semantic Versioning" or semver) which 
uses a three-component number ``A.B.C``

where

- **A** = *Major* version  
  A change to the major version indicates that **breaking changes** to the stable, public API may 
  have been introduced. Users should expect that code written for version `A.x.y` may need 
  modification if upgrading to version `A+1.0.0`.

- **B** = *Minor* version  
  A change to the minor version indicates that **new features** or **enhancements** have been added 
  **without** breaking the stable API, or that some APIs have been deprecated (marked for future 
  removal) but still behave as before (possibly with warnings).

- **C** = *Patch* version  
  A change to the patch version indicates **bug fixes**, **performance improvements**, or minor 
  internal refactors that **preserve** the API and behaviour. Code written for `A.B.C` should 
  continue to work under `A.B.(C+1)`.

### When do we increment version components?

| Component | When it is increased                    | What you should expect                                                                            |
|-----------|----------------------------------------|---------------------------------------------------------------------------------------------------|
| Major (A) | When we introduce changes that break the stable API: e.g., remove or rename public methods, change method signatures, change behaviour in a way that could make existing code fail. | Upgrading across major versions may require code changes. Major versions are released rarely.     |
| Minor (B) | When we add new features, modules or enhancements, OR when we mark existing public APIs as deprecated (but still functional) without breaking their current behaviour. | Upgrading from `A.(B).C` -> `A.(B+1).0` is usually safe, but keep an eye on deprecation warnings. |
| Patch (C) | When we fix bugs, improve stability or performance, or do small internal changes that do *not* affect public API or behaviour. | Upgrading from `A.B.(C)` -> `A.B.(C+1)` should be safe and require no changes.                    |

API Stability & Public vs Private APIs
-------------------------------------

### Public API

We define the *public API* as those classes, functions, modules and methods that are documented and 
whose names do *not* begin with an underscore. Users of neuraLQX should rely exclusively on this 
public API when developing applications or libraries.

We guarantee that **within a given major version** (`A.x.y`), the public API will remain compatible. 
For example, if you write code using neuraLQX version `2.3.4`, it should continue working with 
versions `2.4.0`, `2.5.3`, etc (unless otherwise noted in the Deprecation or Breaking Changes notes in 
the version's Release Notes). By contrast, upgrading to `3.0.0` may require changes.

### Private or Experimental APIs

Methods, classes or modules whose name begins with an underscore (e.g., `_internal_helper`) or which 
reside in modules marked as `experimental` are considered *unstable*. These may change or be removed 
at any time, including within minor or patch version releases. If your code depends on them, you do 
so at your own risk.

### Guidelines for users

- Prefer using **public API** elements: documented modules, classes and functions with no leading underscore.  
- Avoid depending on `experimental` modules unless you explicitly opt-in (via configuration, environment variable, etc.).  
- If you must use private or experimental parts, pin your version (e.g., `==2.4.*`) and be prepared for breaks when upgrading.  
- Keep an eye on **deprecation warnings**, which indicate that something will change in a future minor or major version.

Deprecation Policy
------------------

When we decide to change or remove part of the public API, we follow this policy:

1. A feature or method that will be removed or renamed will first be marked as *deprecated* in a 
   minor version (B+1) or perhaps even in the same minor version if urgent.  
2. Deprecation means the feature still works, but emits a warning (to stdout or logs).  
3. Compatibility remains guaranteed for the remainder of the current major version.  
4. Removal or change happens when we increment the *major* or *minor* version.

Lifecycle and Frequency of Releases
-----------------------------------

- **Patch releases**: frequent, whenever critical bugs, security issues or performance regressions are found.  
- **Minor releases**: when we have accumulated sufficient enhancements or new features (typically once every few months).  
- **Major releases**: rare, reserved for when we need to break backward compatibility or make large architectural shifts.  
- Users should plan to stay within major version branches (e.g., `2.x`) for stability, and update minor/patch versions regularly.

Practical Advice for Users
---------------------------

- **Pin your major version**: In production or long-running projects, you might specify `neuraLQX>=2.0,<3.0` to ensure you stay in the same major branch.  
- **Check release notes**: Always read the changelog for the version you are upgrading to. Watch for "Breaking changes" or "Deprecations".  
- **Avoid private APIs**: Code using private members (leading underscores) may fail unexpectedly.  
- **Enable warnings**: Accept deprecation warnings as they are your early signals for upcoming changes.  
- **Test upgrades**: Before upgrading major versions, run your test suite, update dependencies, and review any deprecation / removal announcements.

FAQ
---

**Q: Can I rely on a patch version (A.B.C+1) being safe?**  
**A:** Yes. Patch versions only contain bug-fixes and minor improvements, so your code should work without modification **unless otherwise mentioned**.

**Q: What if I need a new feature introduced in a minor version (A.B+1.0)?**  
**A:** Upgrading should be safe, but check for any deprecation warnings that might affect your code.

**Q: What should I do if I rely on an experimental module?**  
**A:** Pin your version strictly, enable the relevant flag (e.g., environment variable), and track changes in that experimental part.
