#include "physics_gpu.cuh"
#include "physics_gpu.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <vector>
#include <cmath>

#define MAX_COLLISIONS 250000 

// 1
__global__ void DetectCollisionsKernel(
    int numObjects,
    float* cx, float* cy, float* cz,
    float* sx, float* sy, float* sz,
    int* isStatic, int* isVisible,
    int* outIndexA, int* outIndexB,
    float* outNX, float* outNY, float* outNZ,
    float* outPen,
    int* collisionCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numObjects || !isVisible[i] || isStatic[i]) return;

    for (int j = 0; j < numObjects; ++j) {
        if (i == j || !isVisible[j] || !isStatic[j]) continue;

        float dx = cx[j] - cx[i];
        float dy = cy[j] - cy[i];
        float dz = cz[j] - cz[i];

        float overlapX = (sx[i] + sx[j]) - fabsf(dx);
        float overlapY = (sy[i] + sy[j]) - fabsf(dy);
        float overlapZ = (sz[i] + sz[j]) - fabsf(dz);

        if (overlapX > 0.0f && overlapY > 0.0f && overlapZ > 0.0f) {
            int idx = atomicAdd(collisionCount, 1);
            if (idx < MAX_COLLISIONS) {
                outIndexA[idx] = i; outIndexB[idx] = j;
                if (overlapX < overlapY && overlapX < overlapZ) {
                    outPen[idx] = overlapX; outNX[idx] = (dx > 0.0f) ? 1.0f : -1.0f; outNY[idx] = 0.0f; outNZ[idx] = 0.0f;
                }
                else if (overlapY < overlapZ) {
                    outPen[idx] = overlapY; outNX[idx] = 0.0f; outNY[idx] = (dy > 0.0f) ? 1.0f : -1.0f; outNZ[idx] = 0.0f;
                }
                else {
                    outPen[idx] = overlapZ; outNX[idx] = 0.0f; outNY[idx] = 0.0f; outNZ[idx] = (dz > 0.0f) ? 1.0f : -1.0f;
                }
            }
        }
    }

    for (int j = i + 1; j < numObjects; ++j) {
        if (!isVisible[j] || isStatic[j]) continue;

        float dx = cx[j] - cx[i];
        float dy = cy[j] - cy[i];
        float dz = cz[j] - cz[i];

        float overlapX = (sx[i] + sx[j]) - fabsf(dx);
        float overlapY = (sy[i] + sy[j]) - fabsf(dy);
        float overlapZ = (sz[i] + sz[j]) - fabsf(dz);

        if (overlapX > 0.0f && overlapY > 0.0f && overlapZ > 0.0f) {
            int idx = atomicAdd(collisionCount, 1);
            if (idx < MAX_COLLISIONS) {
                outIndexA[idx] = i; outIndexB[idx] = j;
                if (overlapX < overlapY && overlapX < overlapZ) {
                    outPen[idx] = overlapX; outNX[idx] = (dx > 0.0f) ? 1.0f : -1.0f; outNY[idx] = 0.0f; outNZ[idx] = 0.0f;
                }
                else if (overlapY < overlapZ) {
                    outPen[idx] = overlapY; outNX[idx] = 0.0f; outNY[idx] = (dy > 0.0f) ? 1.0f : -1.0f; outNZ[idx] = 0.0f;
                }
                else {
                    outPen[idx] = overlapZ; outNX[idx] = 0.0f; outNY[idx] = 0.0f; outNZ[idx] = (dz > 0.0f) ? 1.0f : -1.0f;
                }
            }
        }
    }
}

namespace physics {

    static float* d_cx = nullptr, * d_cy = nullptr, * d_cz = nullptr;
    static float* d_sx = nullptr, * d_sy = nullptr, * d_sz = nullptr;
    static int* d_isStatic = nullptr, * d_isVisible = nullptr;
    static int* d_outIndexA = nullptr, * d_outIndexB = nullptr, * d_collisionCount = nullptr;
    static float* d_outNX = nullptr, * d_outNY = nullptr, * d_outNZ = nullptr, * d_outPen = nullptr;
    static int currentCapacity = 0;

    // 2
    bool GPUCollisionDetector::Initialize() {
        cudaSetDevice(0);
        m_initialized = true;
        return true;
    }

    // 3
    std::vector<CollisionInfo> GPUCollisionDetector::DetectCollisions(const std::vector<PhysicsObject>& objects) {
        if (!m_initialized) return {};

        int numObjects = static_cast<int>(objects.size());
        if (numObjects == 0) return {};

        if (numObjects > currentCapacity) {
            if (d_cx) {
                cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_cz);
                cudaFree(d_sx); cudaFree(d_sy); cudaFree(d_sz);
                cudaFree(d_isStatic); cudaFree(d_isVisible);
                cudaFree(d_outIndexA); cudaFree(d_outIndexB); cudaFree(d_collisionCount);
                cudaFree(d_outNX); cudaFree(d_outNY); cudaFree(d_outNZ); cudaFree(d_outPen);
            }

            currentCapacity = numObjects + 1000;

            cudaMalloc(&d_cx, currentCapacity * sizeof(float));
            cudaMalloc(&d_cy, currentCapacity * sizeof(float));
            cudaMalloc(&d_cz, currentCapacity * sizeof(float));
            cudaMalloc(&d_sx, currentCapacity * sizeof(float));
            cudaMalloc(&d_sy, currentCapacity * sizeof(float));
            cudaMalloc(&d_sz, currentCapacity * sizeof(float));
            cudaMalloc(&d_isStatic, currentCapacity * sizeof(int));
            cudaMalloc(&d_isVisible, currentCapacity * sizeof(int));

            cudaMalloc(&d_outIndexA, MAX_COLLISIONS * sizeof(int));
            cudaMalloc(&d_outIndexB, MAX_COLLISIONS * sizeof(int));
            cudaMalloc(&d_outNX, MAX_COLLISIONS * sizeof(float));
            cudaMalloc(&d_outNY, MAX_COLLISIONS * sizeof(float));
            cudaMalloc(&d_outNZ, MAX_COLLISIONS * sizeof(float));
            cudaMalloc(&d_outPen, MAX_COLLISIONS * sizeof(float));
            cudaMalloc(&d_collisionCount, sizeof(int));
        }

        std::vector<float> h_cx(numObjects), h_cy(numObjects), h_cz(numObjects);
        std::vector<float> h_sx(numObjects), h_sy(numObjects), h_sz(numObjects);
        std::vector<int> h_isStatic(numObjects), h_isVisible(numObjects);

        for (int i = 0; i < numObjects; ++i) {
            h_cx[i] = objects[i].boundingVolume.center.x;
            h_cy[i] = objects[i].boundingVolume.center.y;
            h_cz[i] = objects[i].boundingVolume.center.z;
            h_sx[i] = objects[i].boundingVolume.sizes.x;
            h_sy[i] = objects[i].boundingVolume.sizes.y;
            h_sz[i] = objects[i].boundingVolume.sizes.z;
            h_isStatic[i] = objects[i].isStatic ? 1 : 0;
            h_isVisible[i] = (objects[i].color.r >= 0.0f) ? 1 : 0;
        }

        cudaMemcpy(d_cx, h_cx.data(), numObjects * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cy, h_cy.data(), numObjects * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cz, h_cz.data(), numObjects * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sx, h_sx.data(), numObjects * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sy, h_sy.data(), numObjects * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sz, h_sz.data(), numObjects * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_isStatic, h_isStatic.data(), numObjects * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_isVisible, h_isVisible.data(), numObjects * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemset(d_collisionCount, 0, sizeof(int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numObjects + threadsPerBlock - 1) / threadsPerBlock;

        DetectCollisionsKernel << <blocksPerGrid, threadsPerBlock >> > (
            numObjects, d_cx, d_cy, d_cz, d_sx, d_sy, d_sz, d_isStatic, d_isVisible,
            d_outIndexA, d_outIndexB, d_outNX, d_outNY, d_outNZ, d_outPen, d_collisionCount
            );

        cudaDeviceSynchronize();

        int hostCollisionCount = 0;
        cudaMemcpy(&hostCollisionCount, d_collisionCount, sizeof(int), cudaMemcpyDeviceToHost);
        hostCollisionCount = std::min(hostCollisionCount, MAX_COLLISIONS);

        std::vector<CollisionInfo> results;
        if (hostCollisionCount > 0) {
            std::vector<int> h_outA(hostCollisionCount), h_outB(hostCollisionCount);
            std::vector<float> h_outNX(hostCollisionCount), h_outNY(hostCollisionCount);
            std::vector<float> h_outNZ(hostCollisionCount), h_outPen(hostCollisionCount);

            cudaMemcpy(h_outA.data(), d_outIndexA, hostCollisionCount * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outB.data(), d_outIndexB, hostCollisionCount * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outNX.data(), d_outNX, hostCollisionCount * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outNY.data(), d_outNY, hostCollisionCount * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outNZ.data(), d_outNZ, hostCollisionCount * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_outPen.data(), d_outPen, hostCollisionCount * sizeof(float), cudaMemcpyDeviceToHost);

            results.reserve(hostCollisionCount);
            for (int i = 0; i < hostCollisionCount; ++i) {
                CollisionInfo info;
                info.isValid = true;
                info.indexA = h_outA[i];
                info.indexB = h_outB[i];
                info.normal = glm::vec3(h_outNX[i], h_outNY[i], h_outNZ[i]);
                info.penetration = h_outPen[i];
                results.push_back(info);
            }
        }

        return results;
    }

} // namespace physics