# Refactoring Policy

## Purpose

The purpose is to make the model easier to use for users.
Since there are no plans to modify the model itself, design changes considering maintainability will not be made.

## Scope

The scope covers code quality issues in the GNS model.
The issues described in initial-issues.md serve as the starting point.

- This repository contains two types of models: GNS and MeshNet
- MeshNet is out of scope for this refactoring

## User-Facing Issues Before Refactoring

- Environment definitions are written in multiple files with inconsistent content
- Shell scripts with unclear purpose scattered in the repository root
- Ambiguous distinction between modules and scripts
- Difficult to extract and use only specific features

## Refactoring Approach

- Do not change algorithm behavior
- Reorganize existing code with the following perspectives
  - Separation by responsibility
  - Clear distinction between modules and scripts
- Make changes incrementally, maintaining functionality at each stage
