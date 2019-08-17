#ifndef _FAST_SLIC_LSC_H
#define _FAST_SLIC_LSC_H
#include "context.h"

namespace fslic {
	class ContextLSC : public ContextRealDist {
	private:
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
}
#endif
