#pragma once

namespace optim::trustregion {
enum class SubproblemStatus { INTERIOR, BOUNDARY, NEGATIVECURVATURE, HARDCASE };
}