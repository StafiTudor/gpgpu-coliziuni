#include "physics_engine.h"
#include "virtual-world-physics/bounding_volume.h"
#include "virtual-world-physics/collision.h"
#include "virtual-world-physics/physics_gpu.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace physics {
void PhysicsEngine::ClearObjects() { m_objects.clear(); }

void PhysicsEngine::Update(float deltaTime) {
  // Reset statistics
  m_stats.detectedCollisions = 0;
  m_stats.objectCount = static_cast<int>(m_objects.size());

  auto startTime = std::chrono::high_resolution_clock::now();

  // Fixed timestep with accumulator
  m_accumulator += deltaTime;
  int subSteps = 0;
  while (m_accumulator >= m_fixedDeltaTime && subSteps < m_maxSubSteps) {
    ApplyGravity(m_fixedDeltaTime);
    for (auto &object : m_objects) {
      object.Integrate(m_fixedDeltaTime);
    }
    if (m_useGPU && m_gpuDetector) {
      try {
        auto collisions = m_gpuDetector->DetectCollisions(m_objects);

        m_stats.detectedCollisions = collisions.size();

        for (const auto &collision : collisions) {
          ResolveCollision(collision.indexA, collision.indexB, collision);
        }
      } catch (const std::exception &ex) {
        std::cerr << "CUDA collision backend failed, falling back to CPU: "
                  << ex.what() << std::endl;
        m_useGPU = false;
        BroadPhase();
        NarrowPhase();
      }
    } else {
      BroadPhase();
      NarrowPhase();
    }

    m_accumulator -= m_fixedDeltaTime;
    subSteps++;
  }
  if (m_accumulator > m_fixedDeltaTime) {
    m_accumulator = 0.0f;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  m_stats.collisionDetectionTime =
      std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void PhysicsEngine::ApplyGravity(float deltaTime) {
  // TODO
    for (auto& object : m_objects) {
        if (!object.isStatic) {
            object.velocity += m_gravity * deltaTime;
        }
    }
}

void PhysicsEngine::BroadPhase() {  
}


std::vector<std::pair<size_t, size_t>>
PhysicsEngine::GetPotentialCollisionPairs() {
  // TODO
    std::vector<std::pair<size_t, size_t>> pairs;

    // verificarea fiecare cu fiecare O(N^2) din Slide 16.
    for (size_t i = 0; i < m_objects.size(); ++i) {
        if (m_objects[i].color.r < 0.0f) continue; // ignorare pereti invizibili

        for (size_t j = i + 1; j < m_objects.size(); ++j) {
            if (m_objects[j].color.r < 0.0f) continue;

            // excludere coliziuni inutile intre doua obiecte statice ( podea vs pereteex)
            if (m_objects[i].isStatic && m_objects[j].isStatic) continue;

            pairs.push_back({ i, j });
        }
    }
    return pairs;
}

void PhysicsEngine::NarrowPhase() {
  // TODO
    // Faza de verificare care stabileste coliziunile reale (Slide 17)
    auto pairs = GetPotentialCollisionPairs();
    for (const auto& pair : pairs) {
        CollisionInfo info = DetectCollision(pair.first, pair.second);
        if (info.isValid) {
            m_stats.detectedCollisions++;
            ResolveCollision(pair.first, pair.second, info);
        }
    }
}

CollisionInfo PhysicsEngine::DetectCollision(size_t indexA, size_t indexB) {
  // TODO
  return ComputeBoxBoxCollision(indexA, indexB, 
                                  m_objects[indexA].boundingVolume, 
                                  m_objects[indexB].boundingVolume);
}

CollisionInfo
PhysicsEngine::ComputeBoxBoxCollision(size_t indexA, size_t indexB,
                                      const bounding_volume_t &boxA,
                                      const bounding_volume_t &boxB) {
  // TODO
    CollisionInfo info;
    info.isValid = false;

    // calculare distanta dintre centre
    glm::vec3 delta = boxB.center - boxA.center;

    // calcare penetrare pe axele X, Y, Z pentru AABB (Slide 13)
    float overlapX = (boxA.sizes.x + boxB.sizes.x) - std::abs(delta.x);
    float overlapY = (boxA.sizes.y + boxB.sizes.y) - std::abs(delta.y);
    float overlapZ = (boxA.sizes.z + boxB.sizes.z) - std::abs(delta.z);

    // coliziune doar daca, cutiile se suprapun pe absolut toate cele 3 axe
    if (overlapX > 0.0f && overlapY > 0.0f && overlapZ > 0.0f) {
        info.isValid = true;
        info.indexA = indexA;
        info.indexB = indexB;

        // cautare axa cu cea mai mica penetrare pentru a sti pe unde trebuie sa iasa cubul
        if (overlapX < overlapY && overlapX < overlapZ) {
            info.penetration = overlapX;
            info.normal = glm::vec3(delta.x > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
        }
        else if (overlapY < overlapZ) {
            info.penetration = overlapY;
            info.normal = glm::vec3(0.0f, delta.y > 0 ? 1.0f : -1.0f, 0.0f);
        }
        else {
            info.penetration = overlapZ;
            info.normal = glm::vec3(0.0f, 0.0f, delta.z > 0 ? 1.0f : -1.0f);
        }
    }
    return info;
}

void PhysicsEngine::ResolveCollision(size_t indexA, size_t indexB,
                                     const CollisionInfo &collision) {
  // TODO
    PhysicsObject& A = m_objects[indexA];
    PhysicsObject& B = m_objects[indexB];

    float invMassA = A.isStatic ? 0.0f : 1.0f / A.mass;
    float invMassB = B.isStatic ? 0.0f : 1.0f / B.mass;
    float sumInvMass = invMassA + invMassB;

    if (sumInvMass == 0.0f) return;

    // 1. separare (Slide 18)
    // imping cele doua cutii de-a lungul axei de lovire pentru a nu se mai suprapune
    const float percent = 0.8f;
    const float slop = 0.01f;
    glm::vec3 correction = (std::max(collision.penetration - slop, 0.0f) / sumInvMass) * percent * collision.normal;

    if (!A.isStatic) {
        A.position -= invMassA * correction;
        A.UpdateBoundingVolume();
    }
    if (!B.isStatic) {
        B.position += invMassB * correction;
        B.UpdateBoundingVolume();
    }

    // 2 Ricosare (Slide 18)
    // aplicare un impuls scalat de elasticitatea fiecarei cutii
    glm::vec3 relativeVelocity = B.velocity - A.velocity;
    float velocityAlongNormal = glm::dot(relativeVelocity, collision.normal);

    if (velocityAlongNormal > 0.0f) return;

    float e = std::min(A.restitution, B.restitution);
    float j = -(1.0f + e) * velocityAlongNormal;
    j /= sumInvMass;

    glm::vec3 impulse = j * collision.normal;

    if (!A.isStatic) A.velocity -= invMassA * impulse;
    if (!B.isStatic) B.velocity += invMassB * impulse;

    // 3 frecare (Slide 18) ---
    // aplicare impuls de frecare tangent pentru a incetini alunecarea
    relativeVelocity = B.velocity - A.velocity;
    glm::vec3 tangent = relativeVelocity - glm::dot(relativeVelocity, collision.normal) * collision.normal;

    if (glm::length(tangent) > 0.0001f) {
        tangent = glm::normalize(tangent);
        float jt = -glm::dot(relativeVelocity, tangent);
        jt /= sumInvMass;

        float mu = std::sqrt(A.friction * B.friction);
        glm::vec3 frictionImpulse;
        if (std::abs(jt) < j * mu) {
            frictionImpulse = jt * tangent;
        }
        else {
            frictionImpulse = -j * mu * tangent;
        }

        if (!A.isStatic) A.velocity -= invMassA * frictionImpulse;
        if (!B.isStatic) B.velocity += invMassB * frictionImpulse;
    }
}

void PhysicsEngine::Init(const glm::vec3 &gravity, bool useGpu) {
  m_gravity = gravity;
  m_useGPU = false;

  delete m_gpuDetector;
  m_gpuDetector = nullptr;

  if (useGpu) {
    m_gpuDetector = new GPUCollisionDetector();
    if (m_gpuDetector->Initialize()) {
      m_useGPU = true;
    } else {
      delete m_gpuDetector;
      m_gpuDetector = nullptr;
      std::cerr << "CUDA collision backend unavailable; using CPU backend."
                << std::endl;
    }
  }
}

PhysicsEngine::~PhysicsEngine() { delete m_gpuDetector; }
} // namespace physics
