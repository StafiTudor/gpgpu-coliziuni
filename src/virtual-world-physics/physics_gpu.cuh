#pragma once

// Define here functions and auxiliaries for GPU physics computations.

#include <glm/glm.hpp>

namespace physics {

    // Varianta light a cubului, pentru VRAM-ul placii video
    struct GPUCollider {
        glm::vec3 center;
        glm::vec3 sizes;
        bool isStatic;
        bool isVisible;
    };

} // namespace physics