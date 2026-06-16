#include <filesystem>
#include <cmath>
#include <algorithm>
#include <array>
#include "pdal/io/private/GDALGrid.hpp"
#include "pdal/private/gdal/Raster.hpp"
#include "render.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

struct Tile{
    double radius;
    Extent bounds;
    Extent bufferedBounds;
    std::string filename;
};

void render(PointSet *pset, const std::string &outDir, const std::string &outputType,
        int tileSize, 
        const std::vector<double> &radiuses, double resolution, 
        int maxTiles, bool force){
    fs::path pOutDir = fs::path(outDir);
    std::vector<double> rads(radiuses);

    if (fs::exists(pOutDir)){
        if (!force) throw std::runtime_error(outDir + " exists (use --force to overwrite results)");
    }else{
        fs::create_directories(pOutDir);
    }

    // Generate tile list
    unsigned int width = static_cast<int>(std::ceil(pset->extent.width() / resolution));
    unsigned int height = static_cast<int>(std::ceil(pset->extent.height() / resolution));
    
    // Set a floor, no matter the resolution parameter
    // (sometimes a wrongly estimated scale of the model can cause the resolution
    // to be set unrealistically low, causing errors)
    const unsigned int RES_FLOOR = 128;

    if (width < RES_FLOOR && height < RES_FLOOR){
        double prev_width = width;
        double prev_height = height;
        
        if (width >= height){
            width = RES_FLOOR;
            height = static_cast<unsigned int>(std::ceil(pset->extent.height() / pset->extent.width() * RES_FLOOR));
        } else {
            width = static_cast<unsigned int>(std::ceil(pset->extent.width() / pset->extent.height() * RES_FLOOR));
            height = RES_FLOOR;
        }

        double floor_ratio = prev_width / width;
        resolution *= floor_ratio;

        for (size_t i = 0; i < rads.size(); i++){
            rads[i] *= floor_ratio;
        }

        std::cout << "Really low resolution DEM requested (" << prev_width << ", " << prev_height << ") will set floor at " << RES_FLOOR << " pixels. Resolution changed to " << resolution << ". The scale of this reconstruction might be off." << std::endl;
    }

    unsigned int numSplitsX = static_cast<int>(std::max<double>(1.0, std::ceil(width / static_cast<double>(tileSize))));
    unsigned int numSplitsY = static_cast<int>(std::max<double>(1.0, std::ceil(height / static_cast<double>(tileSize))));
    
    unsigned int numTiles = numSplitsX * numSplitsY;

    std::cout << "DEM resolution is (" << width << ", " << height << "), max tile size is " << tileSize << ", will split DEM generation into " << numTiles << " tiles" << std::endl;

    if (maxTiles > 0){
        if (numTiles > maxTiles){
            std::cerr << "Max tiles limit exceeded (" << maxTiles << "). This is a strong indicator that the reconstruction failed" << std::endl;
            exit(1);
        }
    }

    double tileBoundsWidth = pset->extent.width() / static_cast<double>(numSplitsX);
    double tileBoundsHeight = pset->extent.height() / static_cast<double>(numSplitsY);

    std::vector<Tile> tiles;

    double minx;
    double maxx;
    double miny;
    double maxy;

    for (const double &r: rads){
        minx = pset->extent.minx;
        for (unsigned int x = 0; x < numSplitsX; x++){
            miny = pset->extent.miny;
            maxx = x == numSplitsX - 1 ?
                            pset->extent.maxx : 
                            minx + tileBoundsWidth;

            for (unsigned int y = 0; y < numSplitsY; y++){
                maxy = y == numSplitsY - 1 ? 
                                pset->extent.maxy : 
                                miny + tileBoundsHeight;

                std::stringstream ss;
                ss << "r" << r << "_x" << x << "_y" << y << ".tif"; 

                Tile t;
                t.filename = (fs::absolute(pOutDir) / ss.str()).string();
                t.bounds.minx = minx;
                t.bounds.maxx = maxx;
                t.bounds.miny = miny;
                t.bounds.maxy = maxy;
                t.radius = r;

                const double buffer = r * 2;
                t.bufferedBounds.minx = t.bounds.minx - buffer;
                t.bufferedBounds.maxx = t.bounds.maxx + buffer;
                t.bufferedBounds.miny = t.bounds.miny - buffer;
                t.bufferedBounds.maxy = t.bounds.maxy + buffer;

                tiles.push_back(t);

                miny = maxy;
            }
            
            minx = maxx;
        }
    }

    // Sort tiles by decreasing radius
    std::sort(tiles.begin(), tiles.end(), 
        [](Tile const &a, Tile const &b) {
            return a.radius < b.radius; 
        });
    
    int outputTypes;
    if (outputType == "max"){
        outputTypes = pdal::GDALGrid::statMax;
    }else if (outputType == "idw"){
        outputTypes = pdal::GDALGrid::statIdw;
    }else{
        throw std::runtime_error("Unsupported output-type: " + outputType);
    }

    #pragma omp parallel for
    for (int i = 0; i < tiles.size(); i++){
        Tile t = tiles[i];
        int r_width = static_cast<int>(std::floor(t.bounds.width() / resolution) + 1);
        int r_height = static_cast<int>(std::floor(t.bounds.height() / resolution) + 1);

        pdal::GDALGrid grid(t.bounds.minx, t.bounds.miny, 
                            r_width, r_height, 
                            resolution, t.radius, outputTypes, 0, 1.0);
        
        for (size_t i = 0; i < pset->size(); i++){
            if (pset->x[i] >= t.bufferedBounds.minx && pset->x[i] <= t.bufferedBounds.maxx &&
                pset->y[i] >= t.bufferedBounds.miny && pset->y[i] <= t.bufferedBounds.maxy){
                grid.addPoint(pset->x[i], pset->y[i], pset->z[i]);
            }
        }

        std::array<double, 6> pixelToPos;

        pixelToPos[0] = t.bounds.minx;
        pixelToPos[1] = resolution;
        pixelToPos[2] = 0;
        pixelToPos[3] = t.bounds.miny + (resolution * r_height);
        pixelToPos[4] = 0;
        pixelToPos[5] = -resolution;

        pdal::gdal::Raster raster(t.filename, "GTiff", pset->srs, pixelToPos);

        grid.finalize();

        double *src = grid.data(outputType);
        double srcNoData = std::numeric_limits<double>::quiet_NaN();

        // Did we actually write anything, or is this an empty tile?
        bool empty = true;
        size_t pxCount = r_width * r_height;
        for (size_t i = 0; i < pxCount; i++){
            if (!isnan(src[i])){
                empty = false;
                break;
            }
        }

        if (!empty){
            pdal::StringList options;

            pdal::gdal::GDALError err = raster.open(r_width, r_height,
                1, pdal::Dimension::Type::Float, -9999, options);

            if (err != pdal::gdal::GDALError::None) throw std::runtime_error(raster.errorMsg());
            int bandNum = 1;
            
            err = raster.writeBand(src, srcNoData, 1, outputType);
            if (err != pdal::gdal::GDALError::None) throw std::runtime_error(raster.errorMsg());
            raster.close();
        }

        #pragma omp critical
        {
            std::cout << fs::path(t.filename).filename().string() << (empty ? " [Empty]" : "") << std::endl;
        }
    }

}
