#include <gtest/gtest.h>
#include <cmath>

#include <iostream>
#include <vector>
#include <cca.h>

TEST(RowSegmentSetTest, set_from_2d_array) {
    cca::label_no_t labels[] {
        0, 0, 0,
        1, 1, 2,
        3, 3, 4,
        5, 6, 5,
    };

    cca::RowSegmentSet rss;
    rss.set_from_2d_array(labels, 4, 3);

    EXPECT_EQ(rss.get_height(), 4);
    EXPECT_EQ(rss.get_width(), 3);
    EXPECT_EQ(rss.size(), 8);
    auto data = rss.get_data();
    auto row_offsets = rss.get_row_offsets();

    EXPECT_EQ(data.size(), rss.size());
    EXPECT_EQ(row_offsets.size(), rss.get_height() + 1);

    EXPECT_EQ(row_offsets[0], 0);
    EXPECT_EQ(row_offsets[1], 1);
    EXPECT_EQ(row_offsets[2], 3);
    EXPECT_EQ(row_offsets[3], 5);
    EXPECT_EQ(row_offsets[4], 8);

    EXPECT_EQ(data[0].label, 0);
    EXPECT_EQ(data[0].x, 0);
    EXPECT_EQ(data[0].x_end, 3);
    EXPECT_EQ(data[1].label, 1);
    EXPECT_EQ(data[1].x, 0);
    EXPECT_EQ(data[1].x_end, 2);
    EXPECT_EQ(data[2].label, 2);
    EXPECT_EQ(data[2].x, 2);
    EXPECT_EQ(data[2].x_end, 3);

    EXPECT_EQ(data[3].label, 3);
    EXPECT_EQ(data[3].x, 0);
    EXPECT_EQ(data[3].x_end, 2);

    EXPECT_EQ(data[4].label, 4);
    EXPECT_EQ(data[4].x, 2);
    EXPECT_EQ(data[4].x_end, 3);

    EXPECT_EQ(data[5].label, 5);
    EXPECT_EQ(data[5].x, 0);
    EXPECT_EQ(data[5].x_end, 1);

    EXPECT_EQ(data[6].label, 6);
    EXPECT_EQ(data[6].x, 1);
    EXPECT_EQ(data[6].x_end, 2);

    EXPECT_EQ(data[7].label, 5);
    EXPECT_EQ(data[7].x, 2);
    EXPECT_EQ(data[7].x_end, 3);
}

TEST(RowSegmentSetTest, set_from_2d_array_empty_width) {
    cca::label_no_t labels[] {};
    cca::RowSegmentSet rss;
    rss.set_from_2d_array(labels, 4, 0);
    auto data = rss.get_data();
    auto row_offsets = rss.get_row_offsets();

    EXPECT_EQ(data.size(), 0);
    EXPECT_EQ(row_offsets.size(), rss.get_height() + 1);
    EXPECT_EQ(row_offsets[0], 0);
    EXPECT_EQ(row_offsets[1], 0);
    EXPECT_EQ(row_offsets[2], 0);
    EXPECT_EQ(row_offsets[3], 0);
    EXPECT_EQ(row_offsets[4], 0);
}

TEST(DisjointSet, assign_disjoint_set) {
    cca::label_no_t labels[] {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0,
        5, 5, 0,
    };
    cca::DisjointSet disjoint_set;
    cca::RowSegmentSet rss;
    rss.set_from_2d_array(labels, 4, 3);
    cca::assign_disjoint_set(rss, disjoint_set);
    EXPECT_EQ(disjoint_set.size, 7);
    EXPECT_EQ(disjoint_set.find_root(1), disjoint_set.find_root(0));
    EXPECT_EQ(disjoint_set.find_root(2), 2);
    EXPECT_EQ(disjoint_set.find_root(3), 0);
    EXPECT_EQ(disjoint_set.find_root(4), 0);
    EXPECT_EQ(disjoint_set.find_root(5), 5);
    EXPECT_EQ(disjoint_set.find_root(6), 0);

    std::unique_ptr<cca::ComponentSet> cc_set { disjoint_set.flatten() };
    EXPECT_EQ(cc_set->num_components, 3);
    EXPECT_EQ(cc_set->component_assignment[0], 0);
    EXPECT_EQ(cc_set->component_assignment[1], 0);
    EXPECT_EQ(cc_set->component_assignment[2], 1);
    EXPECT_EQ(cc_set->component_assignment[3], 0);
    EXPECT_EQ(cc_set->component_assignment[4], 0);
    EXPECT_EQ(cc_set->component_assignment[5], 2);
    EXPECT_EQ(cc_set->component_assignment[6], 0);
    EXPECT_EQ(cc_set->num_component_members[0], 5);
    EXPECT_EQ(cc_set->num_component_members[1], 1);
    EXPECT_EQ(cc_set->num_component_members[2], 1);
    EXPECT_EQ(cc_set->component_leaders[0], 0);
    EXPECT_EQ(cc_set->component_leaders[1], 2);
    EXPECT_EQ(cc_set->component_leaders[2], 5);
}


TEST(DisjointSet, assign_disjoint_set_2) {
    cca::label_no_t labels[] {
        0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 7,
        0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6,
    };
    cca::DisjointSet disjoint_set;
    cca::RowSegmentSet rss;
    rss.set_from_2d_array(labels, 2, 13);
    cca::assign_disjoint_set(rss, disjoint_set);
    std::unique_ptr<cca::ComponentSet> cc_set { disjoint_set.flatten() };

    EXPECT_EQ(cc_set->num_components, 10);
}


TEST(DisjointSet, estimate_component_area) {
    cca::label_no_t labels[] {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0,
        5, 5, 0,
    };
    cca::RowSegmentSet rss;
    rss.set_from_2d_array(labels, 4, 3);
    cca::DisjointSet disjoint_set;
    cca::assign_disjoint_set(rss, disjoint_set);
    std::unique_ptr<cca::ComponentSet> cc_set { disjoint_set.flatten() };

    std::vector<int> area;
    cca::estimate_component_area(rss, *cc_set, area);
    EXPECT_EQ(area.size(), 3);
    EXPECT_EQ(area[0], 9);
    EXPECT_EQ(area[1], 1);
    EXPECT_EQ(area[2], 2);
}


TEST(DisjointSet, unlabeled_adj) {
    cca::label_no_t u = 0xFFFF;
    cca::label_no_t labels[] {
        0, 0, 0, 0, 0,
        1, 1, u, 0, 0,
        1, u, u, u, 4,
        2, 2, u, u, 4,
        2, 3, 3, 3, 3,
    };
    cca::RowSegmentSet rss;
    rss.set_from_2d_array(labels, 5, 5);
    cca::DisjointSet disjoint_set;
    cca::assign_disjoint_set(rss, disjoint_set);
    std::unique_ptr<cca::ComponentSet> cc_set { disjoint_set.flatten() };

    std::vector<int> area;
    cca::estimate_component_area(rss, *cc_set, area);

    std::vector<cca::label_no_t> label_subs;
    cca::unlabeled_adj(rss, *cc_set, area, label_subs);
    EXPECT_EQ(label_subs[2], 0);
}

TEST(ConnectivityEnforcer, to_choose_largest_cc) {
    cca::label_no_t x = 9;
    cca::label_no_t labels[] {
        0, 0, 0, 0, 0,
        1, 1, x, 0, 0,
        1, x, 0, x, 4,
        2, 2, x, x, 4,
        2, 3, 3, 3, 3,
    };
    cca::ConnectivityEnforcer ce(labels, 5, 5, 10, 0);
    ce.execute(labels);

    /*
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cerr << labels[5 * i + j] << " ";
        }
        std::cerr << std::endl;
    }
    */
    EXPECT_EQ(labels[5 * 1 + 2], 0);
    EXPECT_EQ(labels[5 * 2 + 1], 0);
    EXPECT_EQ(labels[5 * 2 + 2], 0);
    EXPECT_EQ(labels[5 * 2 + 3], 9);
    EXPECT_EQ(labels[5 * 3 + 2], 9);
    EXPECT_EQ(labels[5 * 3 + 3], 9);
}



TEST(ConnectivityEnforcer, time_benchmark) {
    // 480 x 640
    std::vector<cca::label_no_t> labels( 480 * 640 );
    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 640; j++) {
            labels[i * 480 + j] = (i + j) / 144;
        }
    }
    cca::ConnectivityEnforcer ce(&labels[0], 480, 640, (640 + 480) / 144 + 1, 0);
    ce.execute(&labels[0]);
}
