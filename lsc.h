#ifndef _FAST_SLIC_LSC_H
#define _FAST_SLIC_LSC_H
#include "context.h"

namespace fslic {
	class ContextLSC : public ContextRealDist {
	protected:
	    std::vector<uint8_t> image_planes[3]; // L, a, b plane (H x W)
	    std::vector<float> image_features[10]; // l1, l2, a1, a2, b1, b2, x1, x2, y1, y2
	    std::vector<float> image_weights;
	    std::vector<float> centroid_features[10]; // l1, l2, a1, a2, b1, b2, x1, x2, y1, y2
	public:
	    using ContextRealDist::ContextRealDist;
	protected:
	    virtual void before_iteration();
		virtual void after_update();
	    virtual void assign_clusters(const Cluster **target_clusters, int size);
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
