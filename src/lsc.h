#ifndef _FAST_SLIC_LSC_H
#define _FAST_SLIC_LSC_H
#include "context.h"

namespace fslic {
	class ContextLSC : public ContextRealDist {
	protected:
		float C_color = 20;
		float* float_memory_pool = nullptr;
		uint8_t* uint8_memory_pool = nullptr;
	    uint8_t* __restrict image_planes[3]; // L, a, b plane (H x W)
	    float* __restrict image_features[10]; // l1, l2, a1, a2, b1, b2, x1, x2, y1, y2
	    float* __restrict image_weights;
	    float* __restrict centroid_features[10]; // l1, l2, a1, a2, b1, b2, x1, x2, y1, y2
	public:
	    using ContextRealDist::ContextRealDist;
		virtual ~ContextLSC();
	protected:
	    virtual void before_iteration();
		virtual void after_update();
	    virtual void assign_clusters(const Cluster **target_clusters, int size);
		virtual void normalize_features(float *__restrict numers[10], float* __restrict weights, int size);
	private:
	    void map_image_into_feature_space();
	    void map_centroids_into_feature_space();
	};
	class ContextLSCBuilderImpl;
	class ContextLSCBuilder {
	private:
		std::unique_ptr<ContextLSCBuilderImpl> impl;
	public:
		ContextLSCBuilder() : ContextLSCBuilder("standard") {};
		ContextLSCBuilder(const char* arch);
		virtual ~ContextLSCBuilder();
		const char** supported_archs();
		bool is_supported_arch();
		const char* get_arch();
		void set_arch(const char*);
		ContextLSC* build(int H, int W, int K, const uint8_t* image, Cluster *clusters);
	};
}
#endif
